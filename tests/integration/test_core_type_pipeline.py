"""WI-5g end to end: a chip of two core types (4-bit and 2-bit cells) and a layer-to-type rule runs a lane program
through PipelineSession to atol-0 value certificates; every layer quantizes on its type's grid, every core packs onto
its type, and the record carries each type's width and the payload of each core at its own width."""

from __future__ import annotations

import glob
import json
import os

import pytest

from conftest import MockDataProviderFactory

from mimarsinan.pipelining.session import PipelineSession
from integration.test_lane_value_pipeline import MEMBERS, SHAPE, _config

#: The b4442 pattern on the three-stage family: the last stage and the classifier on the 2-bit type.
RULE = [{"layers": ["stages_2_*", "classifier"], "weight_bits": 2}]


def _typed_config(tmp_path, rule=RULE):
    config = _config(tmp_path, *MEMBERS["resnet"][:2])
    platform = config["platform_constraints"]
    core = {"max_axons": 1024, "max_neurons": 256, "count": 4000, "has_bias": True}
    platform["cores"] = [dict(core, weight_bits=4), dict(core, weight_bits=2)]
    platform.update(weight_scale_rule="fractional", bias_resolution="law", layer_core_types=rule)
    return config


@pytest.mark.slow
@pytest.mark.timeout(1500)
def test_a_two_type_chip_certifies_at_atol_zero_and_records_each_type_width(tmp_path, capsys):
    session = PipelineSession.from_config(
        _typed_config(tmp_path),
        data_provider_factory=MockDataProviderFactory(input_shape=SHAPE, num_classes=10, size=64),
    )
    session.run()
    out = capsys.readouterr().out
    assert "[ValueParityGate] PASS model≡identity (max|delta|=0.000e+00, atol=0.0e+00" in out
    certificate = next(line for line in out.splitlines() if line.startswith("[ValueTwinCertificate] PASS"))
    assert "max|delta|=0.000e+00 atol=0.0e+00 passed=True" in certificate and "windows=0 " not in certificate
    paths = glob.glob(os.path.join(str(tmp_path), "**", "deployment_record.json"), recursive=True)
    assert paths, "the pipeline sealed no deployment record"
    record = json.load(open(paths[0], encoding="utf-8"))
    widths = [core["weight_bits"] for core in record["identity"]["platform"]["cores"]]
    assert widths == [4, 2]
    assert record["identity"]["platform"]["layer_core_types"] == RULE
    crossbar = record["utilization"]["crossbar"]
    assert crossbar["programming_bits"] < crossbar["cells_used"] * 4, "2-bit cores program 2 bits per cell"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_a_stale_rule_is_refused_by_name_before_any_core_is_built(tmp_path):
    from mimarsinan.mapping.packing.type_binding import CoreTypeBindingError

    session = PipelineSession.from_config(
        _typed_config(tmp_path, rule=[{"layers": ["no_such_layer"], "weight_bits": 2}]),
        data_provider_factory=MockDataProviderFactory(input_shape=SHAPE, num_classes=10, size=64),
    )
    with pytest.raises(CoreTypeBindingError, match="matches no layer"):
        session.run()
