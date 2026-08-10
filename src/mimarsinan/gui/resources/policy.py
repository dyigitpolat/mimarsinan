"""The one policy deciding WHEN a run's GUI resources are rendered."""

from __future__ import annotations

from enum import Enum

from mimarsinan.common.env import gui_resource_render_override


class ResourceRenderPolicy(str, Enum):
    """Who is watching a step's heavy resources as they are produced.

    Both policies persist each resource's SOURCE data and its cheap
    UI-resolution artifact at snapshot-persist time (the pure-numpy renderer
    made the artifact affordable, and pre-rendering is what makes a browser
    attach a plain file read). ``EAGER`` -- a monitor is attached, so the run
    additionally warms the live in-memory store the monitor serves from.
    ``DEFERRED`` -- nobody is watching, so the store is left cold. Persist
    time never pays for a full-resolution render under either policy: the
    matplotlib-era 1024px backlog once kept a headless process, and under a
    scheduler its whole node, alive for minutes past its last step.
    """

    EAGER = "eager"
    DEFERRED = "deferred"


def resolve_resource_render_policy(declared: ResourceRenderPolicy) -> ResourceRenderPolicy:
    """``declared`` unless an operator override names the other policy; raises on a bad name.

    The run mode declares the policy (``--ui`` renders, ``--headless`` defers);
    the environment override exists so a headless run can be told to render
    in-line anyway, and is the only thing allowed to contradict the declaration.
    """
    override = gui_resource_render_override()
    if override is None:
        return declared
    return ResourceRenderPolicy(override)


__all__ = ["ResourceRenderPolicy", "resolve_resource_render_policy"]
