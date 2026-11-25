""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Mapping, Optional


def refusal_names() -> set:
    ""
    from mimarsinan.common.refusal import ToolchainRefusal

    names, todo = set(), [ToolchainRefusal]
    while todo:
        c = todo.pop()
        if c.__name__ not in names:
            names.add(c.__name__)
            todo += c.__subclasses__()
    return names


def unnamed(detail: str) -> bool:
    ""
    return bool(detail) and detail.split(":", 1)[0] not in refusal_names()


def _absent(key: Any, mapping: Mapping) -> bool:
    try:
        return key not in mapping
    except TypeError:
        return False


def _record(exc: BaseException, path: str, code, f_locals: Mapping, line: int, pkg: str) -> Dict[str, Any]:
    ""
    import torch

    from mimarsinan.common.refusal import ToolchainRefusal

    named = isinstance(exc, ToolchainRefusal)
    where: Dict[str, Any] = {"name": type(exc).__name__, "named": bool(named), "file": os.path.relpath(path, pkg),
                             "function": code.co_name, "line": int(line)}
    if not named:
        params = code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]
        where["empty_inputs"] = sorted(n for n in params if isinstance(f_locals.get(n), torch.Tensor)
                                       and f_locals[n].numel() == 0)
        if isinstance(exc, KeyError) and exc.args:
            absent = sorted(n for n in params if isinstance(f_locals.get(n), Mapping) and _absent(exc.args[0], f_locals[n]))
            if absent:
                where["absent_key_in"] = absent
    return where


def _pkg() -> str:
    import mimarsinan

    return os.path.dirname(os.path.abspath(mimarsinan.__file__)) + os.sep


def site(again: Callable[[], Any], detail: str) -> Dict[str, Any]:
    ""
    pkg = _pkg()
    first: Dict[int, Any] = {}

    def local(frame, event, arg):
        if event == "exception":
            exc = arg[1]
            path = os.path.abspath(frame.f_code.co_filename)
            if id(exc) not in first and path.startswith(pkg):
                first[id(exc)] = (exc, _record(exc, path, frame.f_code, frame.f_locals, frame.f_lineno, pkg))
        return local

    def calls(frame, event, arg):
        frame.f_trace_lines = False
        return local

    sys.settrace(calls)
    try:
        again()
    finally:
        sys.settrace(None)
    for exc, where in first.values():
        if f"{type(exc).__name__}: {str(exc)[:300]}" == detail:
            return where
    return {"reproduced": False}


def switch(control: Optional[Callable[[], Any]], detail: str) -> Optional[Dict[str, Any]]:
    ""
    if control is None:
        return None
    verdict = control()
    return {"released": verdict.outcome, "same_error": bool(verdict.detail) and verdict.detail == detail}


def raise_site(exc: BaseException) -> Dict[str, Any]:
    ""
    pkg = _pkg()
    tb, found = exc.__traceback__, None
    while tb is not None:
        path = os.path.abspath(tb.tb_frame.f_code.co_filename)
        if path.startswith(pkg):
            found = (path, tb.tb_frame, tb.tb_lineno)
        tb = tb.tb_next
    if found is None:
        return {"reproduced": False}
    path, frame, line = found
    return _record(exc, path, frame.f_code, frame.f_locals, line, pkg)
