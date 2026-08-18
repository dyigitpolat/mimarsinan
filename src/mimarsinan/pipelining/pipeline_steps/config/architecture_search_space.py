"""The MODEL-side search space of one run: its axes and its raw->config assembler.

A builder declares what it can vary (``get_nas_search_options``, plus every
multi-valued ``select`` in its config schema) and the run's ``arch_search``
narrows it; nothing here knows about optimizers, which is why it sits beside
the factory rather than inside it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def derive_arch_options(
    builder_cls: type,
    arch_cfg: Dict[str, Any],
    input_shape: tuple,
) -> Tuple[List[Tuple[str, List[Any]]], Dict[str, Any]]:
    schema = getattr(builder_cls, "get_config_schema", lambda: [])()
    schema_map = {f["key"]: f for f in schema}

    nas_opts_fn = getattr(builder_cls, "get_nas_search_options", None)
    builder_options: Dict[str, List[Any]] = (
        nas_opts_fn(input_shape=input_shape) if nas_opts_fn else {}
    )

    arch_options: List[Tuple[str, List[Any]]] = []
    for field_desc in schema:
        key = field_desc["key"]
        field_type = field_desc.get("type")
        if field_type == "select" and "options" in field_desc:
            values = arch_cfg.get(f"{key}_options", field_desc["options"])
            if len(values) > 1:
                arch_options.append((key, list(values)))
        elif key in builder_options:
            values = arch_cfg.get(f"{key}_options", builder_options[key])
            if len(values) > 1:
                arch_options.append((key, list(values)))

    schema_keys = {f["key"] for f in schema}
    for key, default_values in builder_options.items():
        if key not in schema_keys:
            values = arch_cfg.get(f"{key}_options", default_values)
            if len(values) > 1:
                arch_options.append((key, list(values)))

    return arch_options, schema_map


def make_assembler(schema: List[Dict[str, Any]], schema_map: Dict[str, Any]):
    def assembler(raw: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for field_desc in schema:
            if "default" in field_desc:
                result[field_desc["key"]] = field_desc["default"]
        for k, v in raw.items():
            field_info = schema_map.get(k, {})
            if field_info.get("type") == "number":
                default = field_info.get("default", 0)
                result[k] = float(v) if isinstance(default, float) else int(v)
            else:
                result[k] = v
        return result
    return assembler


def resolve_arch_options(
    builder_cls, arch_cfg: Dict, input_shape: Tuple, *,
    searches_model: bool, model_type: str,
):
    """The model-side search space and its raw->config assembler for one run.

    A hardware-only search has neither: its model config is fixed, so the
    options list is empty and the assembler is the identity.
    """
    if not searches_model:
        return [], (lambda raw: dict(raw))

    arch_options, schema_map = derive_arch_options(builder_cls, arch_cfg, input_shape)
    schema = getattr(builder_cls, "get_config_schema", lambda: [])()
    if not arch_options:
        raise NotImplementedError(
            f"No NAS search space defined for model_type='{model_type}'. "
            f"Add get_nas_search_options() or 'select' fields with multiple options "
            f"to {builder_cls.__name__}.get_config_schema(). "
            f"Current schema keys: {[f['key'] for f in schema]}"
        )
    return arch_options, make_assembler(schema, schema_map)
