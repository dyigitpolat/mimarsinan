"""E282-E284 end to end: the integer readout fixture runs through PipelineSession to atol-0 value certificates."""

from __future__ import annotations

import json
import os
import re

import pytest

from conftest import MockDataProviderFactory

from mimarsinan.pipelining.session import PipelineSession

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "integer_value", "mvm_int_readout_lenet5.json")


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_integer_readout_fixture_certifies_at_atol_zero(tmp_path, capsys):
    config = json.load(open(FIXTURE, encoding="utf-8"))
    config["generated_files_path"] = str(tmp_path)
    config["deployment_parameters"].update({
        # Hermetic data and a short schedule; the declared program (laws, grids, passes) is the fixture's.
        "device": "cpu", "num_workers": 0, "batch_size": 16, "training_epochs": 1, "tuning_budget_scale": 0.05,
        "pretrain_floor_chance_multiple": 0, "degradation_tolerance": 0.9,
    })
    session = PipelineSession.from_config(
        config, data_provider_factory=MockDataProviderFactory(input_shape=(1, 28, 28), num_classes=10, size=64),
    )
    session.run()
    out = capsys.readouterr().out
    assert "Running 'Readout Law'" in out
    assert "[ValueParityGate] PASS model≡identity (max|delta|=0.000e+00, atol=0.0e+00" in out
    certificate = next(line for line in out.splitlines() if line.startswith("[ValueTwinCertificate] PASS"))
    assert "max|delta|=0.000e+00 atol=0.0e+00 passed=True" in certificate
    assert "windows=0 " not in certificate
    program = next(line for line in out.splitlines() if "Hybrid program (scheduled)" in line)
    passes = [int(count) for count in re.findall(r"seg \d+: (\d+) pass", program)]
    assert passes and max(passes) >= 2, program  # a segment runs in more than one pass
