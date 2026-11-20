"""[U-CHIPKEYS] Hand-made test chips under the loader rule: a test chip states its core semantics and names every
other key that changes the computation unstated (the test's own), never leaving one to a registry default."""

import copy

from mimarsinan.mapping.platform.descriptions.computation_keys import skeleton
from mimarsinan.mapping.platform.target_bundles import TargetBundleError, bundle_values

NOTE = "a hand-made test chip: the test declares what it needs"


def described(data, semantics="mvm", note=NOTE):
    """``data`` (a bundle file's object) with ``core_semantics`` (``semantics`` unless its keys set one) and a
    computation block naming every computation key unstated; a bundle whose keys the loader refuses is returned
    without a block, so the test reads that refusal."""
    data = copy.deepcopy(data)
    keys = data.setdefault("keys", {})
    if isinstance(keys, dict):
        keys.setdefault("core_semantics", semantics)
    try:
        _, _, values = bundle_values(f"{data.get('name', 'hand')}.json", data)
    except (TargetBundleError, KeyError, TypeError):
        return data
    data["computation"] = skeleton(values, note)
    return data
