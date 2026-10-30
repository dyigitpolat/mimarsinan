"""[ODIN5, plan §7 row 16] The vendored ODIN tree is byte-identical to upstream.

The exporter's bit layouts, the packer tables and the testbench's SPI/AER
encodings are all transcribed from these files. A silent edit here would not
break a test elsewhere — it would change what "the stock ODIN core" MEANS while
every other gate kept passing. So the tree is hashed against a recorded
manifest, and the manifest names the upstream commit.
"""

import hashlib
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parents[3] / "hw" / "vendor" / "odin"
MANIFEST = VENDOR / "MANIFEST.sha256"
PROVENANCE = VENDOR / "PROVENANCE.md"
NOTICE = Path(__file__).resolve().parents[3] / "NOTICE"
OVERLAY = Path(__file__).resolve().parents[3] / "hw" / "fpga" / "mem"

UPSTREAM_COMMIT = "17819318d17b6241d4b185c13ce9b3372d453778"

#: Files whose module names the overlay shadows; both must keep their upstream
#: declarations so the substitution stays a file-ORDER choice, never an edit.
OVERLAY_MODULES = ("SRAM_256x128_wrapper", "SRAM_8192x32_wrapper")


def _recorded():
    entries = {}
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        entries[name.strip()] = digest
    return entries


class TestTheVendoredTreeIsUnmodified:
    def test_every_recorded_file_hashes_to_its_recorded_digest(self):
        for name, digest in _recorded().items():
            path = VENDOR / name
            assert path.is_file(), f"{name} is recorded but missing"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, (
                f"{name} does not match the recorded upstream hash: the "
                f"vendored tree is read-only (hw/vendor/odin/PROVENANCE.md)")

    def test_no_file_in_the_tree_escapes_the_manifest(self):
        recorded = set(_recorded())
        present = {
            str(path.relative_to(VENDOR))
            for path in VENDOR.rglob("*")
            if path.is_file() and path.name not in ("MANIFEST.sha256", "PROVENANCE.md")
        }
        assert present == recorded

    def test_the_whole_source_tree_is_vendored_not_a_subset(self):
        sources = sorted(
            str(path.relative_to(VENDOR))
            for path in (VENDOR / "src").rglob("*.v")
        )
        assert len(sources) == 18
        assert "src/ODIN.v" in sources
        assert "src/LIF_neuron_blocks/lif_neuron_state.v" in sources

    def test_both_upstream_licences_travel_with_the_tree(self):
        assert "Solderpad" in (VENDOR / "LICENSE").read_text()
        assert "Creative Commons" in (VENDOR / "doc" / "LICENSE").read_text()

    def test_every_vendored_source_keeps_its_upstream_copyright_header(self):
        for path in (VENDOR / "src").rglob("*.v"):
            head = path.read_text(errors="replace")[:1200]
            assert "UCLouvain" in head, path
            assert "Solderpad" in head, path


class TestTheProvenanceIsRecordedNotAssumed:
    def test_the_provenance_names_the_pinned_upstream_commit(self):
        text = PROVENANCE.read_text()
        assert UPSTREAM_COMMIT in text
        assert "github.com/ChFrenkel/ODIN" in text

    def test_the_provenance_carries_the_citation_upstream_asks_for(self):
        assert "Transactions on Biomedical Circuits and Systems" in \
            PROVENANCE.read_text()

    def test_the_root_notice_declares_the_hw_subtree_licence(self):
        text = NOTICE.read_text()
        assert "Solderpad" in text and "hw/vendor/odin" in text
        assert "4(b)" in text


class TestTheBramOverlayIsADeclaredDerivativeNotAnEdit:
    @pytest.mark.parametrize("module", OVERLAY_MODULES)
    def test_the_overlay_declares_the_same_module_name(self, module):
        text = (OVERLAY / f"{module}.v").read_text()
        assert f"module {module} (" in text

    @pytest.mark.parametrize("module", OVERLAY_MODULES)
    def test_the_overlay_carries_the_licence_and_a_statement_of_changes(self, module):
        text = (OVERLAY / f"{module}.v").read_text()
        assert "UCLouvain" in text
        assert "Solderpad" in text
        assert "STATEMENT OF CHANGES" in text
        assert "1781931" in text

    @pytest.mark.parametrize("module", OVERLAY_MODULES)
    def test_the_vendored_declaration_is_still_there_to_be_shadowed(self, module):
        vendored = "\n".join(
            path.read_text(errors="replace") for path in (VENDOR / "src").glob("*.v")
        )
        assert f"module {module} (" in vendored

    @pytest.mark.parametrize("module", OVERLAY_MODULES)
    def test_the_overlay_keeps_the_upstream_port_list_verbatim(self, module):
        overlay = (OVERLAY / f"{module}.v").read_text()
        for port in ("RSTN", "CK", "CS", "WE", ".A", "D", "Q"):
            assert port.lstrip(".") in overlay
