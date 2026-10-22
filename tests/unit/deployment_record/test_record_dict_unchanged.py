"""No format_version change and no G3 record field: a record built without an executed residency plan
keeps its bytes. The digests were taken on the Wave 1 base (w1-base-s0) before G3's changes."""

import hashlib
import json

import pytest

from mimarsinan.deployment_record.build.from_mapping import (
    placement_record_from_mapping,
    schedule_record_from_mapping,
)
from mimarsinan.deployment_record.schema import DeploymentRecord
from mimarsinan.deployment_record.schema.record import DEPLOYMENT_RECORD_FORMAT_VERSION

from unit.deployment_record.record_fixtures import make_full_record
from unit.deployment_record.g3_fixtures import built_chain

#: (schedule, placement) to_dict digests of the capacity-split chain at 1 and 3 cores, on the base.
_BASE_BUILD_DIGESTS = {
    1: ("d096179af3a3b3d823c1425b2cb10892e6977cea756a97b834fa65ae0047c767",
        "9b30d3b8122a29f97f60ef8e08775ebf3da81a46d7ccaee30555f4a40271d328"),
    3: ("0d7166dea2953169b102ecaa950b7100fb064030d5cfb29e2ee8cc58f4222740",
        "1c118b3ea8b0fa6af475a9f7a0df5a391a034735b405a0508456d2cbd9a81480"),
}
_BASE_FULL_RECORD_DIGEST = "401f969c9f01ea108e15e1a473d540a9fe9805af100159a5646b84b9bd55a494"


def _digest(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def test_the_format_version_is_unchanged():
    assert DEPLOYMENT_RECORD_FORMAT_VERSION == 1


def test_a_wave_zero_record_keeps_its_bytes_and_loads():
    record = make_full_record()
    assert _digest(record.to_dict()) == _BASE_FULL_RECORD_DIGEST
    assert DeploymentRecord.from_dict(record.to_dict()) == record


@pytest.mark.parametrize("count", sorted(_BASE_BUILD_DIGESTS))
def test_the_record_build_is_byte_identical_after_the_census_split(count):
    mapping = built_chain(count)
    schedule = schedule_record_from_mapping(mapping, weight_bits=8, params_reloaded=0, timesteps=4,
                                            pass_transfer="collapse").to_dict()
    placement = placement_record_from_mapping(mapping).to_dict()
    assert (_digest(schedule), _digest(placement)) == _BASE_BUILD_DIGESTS[count]
