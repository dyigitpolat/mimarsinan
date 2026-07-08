"""Dev-mode workbench review: drive the real configurator in a real browser and
save the round-6 evidence set to generated/_wizard_review/round6/:
fresh per-section shots, the co-design interaction sequence (mapping updating
while arch + grid are edited), the mapping-strategy panel ABOVE the full-width
mapping panel, the merged side-by-side Training & Tuning section with the
mirror mode, green derived-value rendering, the vehicle toggles staying live
under unrelated errors, float-weights authoring, each of the five mode
switches, the template flow, and the error/remedy flow.

Usage (from the repo root, venv active; playwright + chromium required):
    python scripts/wizard_screenshots.py [--out DIR]

NOT part of the test suite: it needs a browser and binds a local port.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SECTION_IDS = ("workload", "codesign", "semantics", "training", "review")
TEMPLATE_TIER_CONFIG = REPO_ROOT / "test_configs" / "tier0" / "t0_02_lif_lenet5_fp_s8_novena_offload_pruned.json"

MODE_SWITCHES = (
    ("lif", None),
    ("ttfs", None),
    ("ttfs_quantized", None),
    ("ttfs_cycle_based", "cascaded"),
    ("ttfs_cycle_based", "synchronized"),
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int) -> None:
    import uvicorn

    from mimarsinan.gui.runtime.collector import DataCollector
    from mimarsinan.gui.server.app import create_app

    app = create_app(DataCollector())
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("GUI server did not come up")


def _settle(page, ms: int = 1200) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def _goto_section(page, section_id: str) -> None:
    page.click(f'.wb-nav-item[data-section-id="{section_id}"]')
    _settle(page, 600)


def _click_segment(page, key: str, option: str) -> None:
    page.click(f'.field[data-key="{key}"] .seg-btn:text-is("{option}")')
    _settle(page, 900)


def _shot(page, out_dir: Path, name: str, shots: list[str], full: bool = False) -> None:
    path = out_dir / f"{name}.png"
    page.screenshot(path=str(path), full_page=full)
    shots.append(str(path.relative_to(REPO_ROOT)))


def _shot_element(locator, out_dir: Path, name: str, shots: list[str]) -> None:
    path = out_dir / f"{name}.png"
    locator.screenshot(path=str(path))
    shots.append(str(path.relative_to(REPO_ROOT)))


def _shoot_sections(page, base_url: str, out_dir: Path, flow: str,
                    shots: list[str], query: str = "") -> None:
    page.goto(base_url + "/wizard" + query)
    _settle(page, 2200)
    for i, section_id in enumerate(SECTION_IDS, start=1):
        _goto_section(page, section_id)
        _shot(page, out_dir, f"{flow}_{i}_{section_id}", shots, full=True)


def _shoot_codesign_sequence(page, base_url: str, out_dir: Path, shots: list[str]) -> None:
    """The J2 loop on one screen: edit grid + arch, watch the mapping re-plan."""
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "codesign")
    _shot(page, out_dir, "codesign_1_initial", shots, full=True)

    # Shrink the first core type's count: the plan reacts (waste/feasibility).
    count_box = page.locator(".cores-editor-row").first.locator("input").nth(2)
    count_box.fill("4")
    count_box.dispatch_event("change")
    _settle(page, 1600)
    _shot(page, out_dir, "codesign_2_grid_edit_mapping_updates", shots, full=True)

    # Suggest hardware for this model: the grid rewrites, mapping re-plans.
    page.click(".suggest-hw")
    _settle(page, 2000)
    _shot(page, out_dir, "codesign_3_suggest_hardware", shots, full=True)

    # Architecture edit on the same screen: another vehicle, live re-plan.
    page.select_option('.field[data-key="model_type"] select', "simple_mlp")
    _settle(page, 1800)
    _shot(page, out_dir, "codesign_4_arch_edit_mapping_updates", shots, full=True)


def _shoot_cosearch_and_placeholders(page, base_url: str, out_dir: Path,
                                     shots: list[str]) -> None:
    """User-amendment evidence: the co-search panel is its OWN card spanning
    model and hardware (off → quiet affordance; active → primary fields with
    ownership chips on the hand cards), and empty semantics render as faded
    in-field placeholders (no under-field hint lines)."""
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "codesign")
    # Both config modes hand-specified: one quiet off affordance, zero
    # search fields anywhere on the screen.
    _shot(page, out_dir, "cosearch_1_off", shots, full=True)

    # The off card's derived enable path activates model search: the panel
    # renders its keys primary, sibling to Model and Hardware.
    page.click('.section-off .btn-sm:text-is("Model Config Mode → search")')
    _settle(page, 1400)
    _shot(page, out_dir, "cosearch_2_model_search_active", shots, full=True)

    # Model-card honesty: the hand field slot states that the co-search owns
    # model_config now.
    _shot_element(page.locator('.section[data-section="model"]'),
                  out_dir, "cosearch_3_model_card_search_owned", shots)

    # Hardware search too: the core-grid hand editor gives way to its
    # ownership chip and the HW search space joins the co-search panel.
    _click_segment(page, "hw_config_mode", "search")
    _shot(page, out_dir, "cosearch_4_joint_search_hw_owned", shots, full=True)

    # Placeholder discipline: the empty behavior lives INSIDE the input as
    # faded blue text (default value or derived phrase), no hint lines.
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "training")
    page.locator('.wb-section[data-section-id="training"] .advanced-toggle').first.click()
    _settle(page, 500)
    _shot(page, out_dir, "placeholders_1_tuning_advanced", shots, full=True)


def _shoot_defect_evidence(page, base_url: str, out_dir: Path, shots: list[str]) -> None:
    """Targeted evidence: search-mode prominence, the pruning slider knob,
    and the in-field placeholder discipline inside an Advanced drawer."""
    page.goto(base_url + "/wizard")
    _settle(page, 2200)

    # Defect 1: =search promotes the NAS panel; model_type/model_config vanish.
    _goto_section(page, "codesign")
    _click_segment(page, "model_config_mode", "search")
    _shot(page, out_dir, "defect1_search_mode_promotes_nas", shots, full=True)
    _click_segment(page, "model_config_mode", "user")

    # Defect 3 (round-2) + round-3 item 5: pruning is a mapping-strategy
    # choice on the Co-Design panel and opens a slider knob there.
    _goto_section(page, "codesign")
    page.click('.field[data-key="pruning"] .toggle-row')
    _settle(page, 900)
    _shot(page, out_dir, "defect3_pruning_slider_in_mapping_strategy", shots, full=True)
    page.click('.field[data-key="pruning"] .toggle-row')
    _settle(page, 600)

    # Defect 2: empty boxes state what empty means (Advanced drawer).
    _goto_section(page, "training")
    page.locator('.wb-section[data-section-id="training"] .advanced-toggle').first.click()
    _settle(page, 500)
    _shot(page, out_dir, "defect2_empty_hints_advanced_drawer", shots, full=True)


def _shoot_round3_evidence(page, base_url: str, out_dir: Path, shots: list[str]) -> None:
    """Round-3 defect evidence: mapping-strategy panel, float authoring,
    vehicle toggles + co-located settings, 3-col co-search + structured
    search space, cores editor, immediate tooltips."""
    page.goto(base_url + "/wizard")
    _settle(page, 2200)

    # Round-4 item 1: the mapping-strategy panel sits ABOVE the live mapping
    # panel; the mapping panel spans the full co-design width.
    _goto_section(page, "codesign")
    _shot_element(page.locator(".codesign-lower"), out_dir,
                  "mapping_strategy_1_panel_above_fullwidth_mapping", shots)
    page.click('.field[data-key="allow_scheduling"] .toggle-row')
    _settle(page, 1400)
    _shot(page, out_dir, "mapping_strategy_2_strategy_edit_replans", shots, full=True)
    page.click('.field[data-key="allow_scheduling"] .toggle-row')
    _settle(page, 600)

    # Item 2: the modernized core-types editor.
    _shot_element(page.locator('.section[data-section="hardware"]'), out_dir,
                  "cores_editor_modernized", shots)

    # Item 3: float weights are first-class; the assembly drops the WQ steps
    # and the emitted document carries the tier-config fp form.
    page.click('.field[data-key="weight_bits"] .seg-btn:text-is("float")')
    _settle(page, 1400)
    _shot(page, out_dir, "float_1_authoring_codesign", shots, full=True)
    _goto_section(page, "review")
    _shot(page, out_dir, "float_2_emitted_vanilla_document", shots, full=True)
    _goto_section(page, "codesign")
    page.click('.field[data-key="weight_bits"] .seg-btn:text-is("quantized")')
    _settle(page, 900)

    # Item 6: supported vehicles are honest toggles with co-located settings;
    # switching one off removes its step and stores an explicit off.
    _goto_section(page, "semantics")
    _shot_element(page.locator("#vehiclesCard"), out_dir,
                  "vehicles_1_toggles_colocated_settings", shots)
    page.locator(".vehicle-row .vehicle-toggle").last.click()
    _settle(page, 1400)
    _shot(page, out_dir, "vehicles_2_user_off_step_removed", shots, full=True)
    _goto_section(page, "review")
    _shot(page, out_dir, "vehicles_3_explicit_off_in_document", shots, full=True)

    # Unsupported stays a muted line (sync mode: nevresim unavailable).
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "semantics")
    _click_segment(page, "spiking_mode", "ttfs_cycle_based")
    _click_segment(page, "ttfs_cycle_schedule", "synchronized")
    _shot_element(page.locator("#vehiclesCard"), out_dir,
                  "vehicles_4_unsupported_muted_line", shots)

    # Item 10: the co-search configurator packs 3 columns and search_space
    # renders as the structured bounds editor.
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "codesign")
    _click_segment(page, "hw_config_mode", "search")
    _settle(page, 900)
    _shot_element(page.locator(".codesign-cosearch"), out_dir,
                  "cosearch_5_three_columns_structured_search_space", shots)

    # Item 9: immediate tooltip — a truncated faded placeholder reveals its
    # full empty semantics on hover, with zero delay.
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "training")
    page.locator('.wb-section[data-section-id="training"] .advanced-toggle').last.click()
    _settle(page, 500)
    page.hover('.field[data-key="eval_subsample_target"] input')
    page.wait_for_timeout(250)
    _shot(page, out_dir, "tooltip_1_immediate_placeholder_reveal", shots, full=True)


def _shoot_round4_evidence(page, base_url: str, out_dir: Path, broken_id: str,
                           shots: list[str]) -> None:
    """Round-4 evidence: the merged side-by-side Training & Tuning section,
    the mirror-training-recipe mode, green derived rendering, and vehicle
    toggles that stay live while unrelated errors are active."""
    page.goto(base_url + "/wizard")
    _settle(page, 2200)

    # Item 6: one Training & Tuning section, panels side-by-side, recipes basic.
    _goto_section(page, "training")
    _shot(page, out_dir, "round4_training_tuning_side_by_side", shots, full=True)

    # Item 6: mirror mode reflects the training recipe; the tuning recipe slot
    # becomes a training-owned ownership chip.
    page.click('.field[data-key="mirror_training_recipe"] .toggle-row')
    _settle(page, 1200)
    _shot(page, out_dir, "round4_mirror_training_recipe_on", shots, full=True)
    page.click('.field[data-key="mirror_training_recipe"] .toggle-row')
    _settle(page, 600)

    # Item 8: green derived rendering — concrete "derived: <value>"
    # placeholders in the tuning Advanced drawer + green derived chips.
    page.locator('.wb-section[data-section-id="training"] .advanced-toggle').last.click()
    _settle(page, 500)
    _shot(page, out_dir, "round4_green_derived_placeholders", shots, full=True)
    _goto_section(page, "review")
    _shot_element(page.locator('[data-section="derived"]'), out_dir,
                  "round4_green_derived_chips", shots)

    # Item 5: vehicle toggles stay live while an unrelated contract error is
    # active (the broken template violates the WQ contract).
    page.goto(base_url + "/wizard?template_id=" + broken_id)
    _settle(page, 2200)
    _goto_section(page, "semantics")
    _shot_element(page.locator("#vehiclesCard"), out_dir,
                  "round4_vehicles_live_under_error_before", shots)
    page.locator(".vehicle-row .vehicle-toggle").last.click()
    _settle(page, 1200)
    _shot_element(page.locator("#vehiclesCard"), out_dir,
                  "round4_vehicles_live_under_error_toggled", shots)
    _shot(page, out_dir, "round4_vehicles_live_under_error_full", shots, full=True)

    # Item 2: encoding-layer placement is an explicit choice (baseline-pinned
    # segmented control on the mapping-strategy panel, no schema default).
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "codesign")
    _shot_element(page.locator(".codesign-strategy"), out_dir,
                  "round4_mapping_strategy_panel", shots)


def _shoot_round5_evidence(page, base_url: str, out_dir: Path,
                           shots: list[str]) -> None:
    """Round-5 evidence: the Model→Mapping-strategy left column beside
    Hardware with the mapping panel full-width beneath; the derived-but-
    overridable spiking knobs with their SSOT provenance; negative_value_shift
    back as a mapping-strategy knob; the deduped builder-provided weight
    source; derived chips only in Review."""
    page.goto(base_url + "/wizard")
    _settle(page, 2200)

    # Item 1: left column = Model then Mapping strategy; right = Hardware;
    # mapping performance spans the full width beneath both.
    _goto_section(page, "codesign")
    _shot(page, out_dir, "round5_1_layout_strategy_below_model", shots, full=True)
    _shot_element(page.locator('.wb-section[data-section-id="codesign"] .wb-cols'),
                  out_dir, "round5_1_layout_columns", shots)

    # Item 2a: negative_value_shift is a mapping-strategy knob again (default
    # on = the calibrated shift; off = the mapper's subsume-forward path).
    _shot_element(page.locator(".codesign-strategy"), out_dir,
                  "round5_2_negative_value_shift_knob", shots)
    page.click('.field[data-key="negative_value_shift"] .toggle-row')
    _settle(page, 1200)
    _shot_element(page.locator(".codesign-strategy"), out_dir,
                  "round5_2_negative_value_shift_off", shots)
    page.click('.field[data-key="negative_value_shift"] .toggle-row')
    _settle(page, 800)

    # Item 2b: firing/spike-generation/thresholding are declarable again and
    # show their derived default in green until an explicit value wins.
    _goto_section(page, "semantics")
    page.locator('.wb-section[data-section-id="semantics"] .advanced-toggle').first.click()
    _settle(page, 600)
    _shot(page, out_dir, "round5_2_spiking_modes_derived_overridable", shots, full=True)
    page.click('.field[data-key="firing_mode"] .seg-btn:text-is("Novena")')
    _settle(page, 1200)
    _shot(page, out_dir, "round5_2_spiking_mode_explicit_wins", shots, full=True)

    # Item 3: every derived-by-default value names its SSOT source.
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "training")
    for toggle in page.locator(
            '.wb-section[data-section-id="training"] .advanced-toggle').all():
        toggle.click()
    _settle(page, 800)
    _shot(page, out_dir, "round5_3_provenance_badges_training_tuning", shots, full=True)

    # Item 4: ONE weight-source concept — preload_weights is the hand knob and
    # the source itself derives from the model builder's registration.
    _shot_element(page.locator('[data-section="training"]'), out_dir,
                  "round5_4_preload_weights_hand_knob", shots)

    # lenet5 registers no pretrained source: the regime fails LOUD (the same
    # error DeploymentPlan raises), keyed to the derived weight_source.
    page.click('.field[data-key="preload_weights"] .toggle-row')
    _settle(page, 1600)
    _shot(page, out_dir, "round5_4_preload_without_registration_is_loud", shots,
          full=True)

    # A builder that DOES register one (torch_vit) resolves the source to a
    # green derived value; no hand field for it exists anywhere.
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "codesign")
    page.select_option('.field[data-key="model_type"] select', "torch_vit")
    _settle(page, 2000)
    _goto_section(page, "training")
    page.click('.field[data-key="preload_weights"] .toggle-row')
    _settle(page, 1600)
    _shot(page, out_dir, "round5_4_builder_provided_weight_source", shots, full=True)
    _goto_section(page, "review")
    _shot_element(page.locator('[data-section="derived"]'), out_dir,
                  "round5_4_weight_source_derived_chip", shots)

    # Item 5: derived chips render ONLY in the Review "Derived values" panel —
    # every other section page stays clean.
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "semantics")
    _shot(page, out_dir, "round5_5_semantics_page_has_no_chips", shots, full=True)
    _goto_section(page, "review")
    _shot_element(page.locator('[data-section="derived"]'), out_dir,
                  "round5_5_derived_values_panel_with_provenance", shots)
    _shot(page, out_dir, "round5_5_review_section", shots, full=True)


def _open_advanced(page, group: str) -> None:
    """Open a group card's Advanced drawer (closed unless a knob is edited)."""
    drawer = page.locator(f'.section[data-section="{group}"] .advanced-drawer')
    if not drawer.first.evaluate("node => node.classList.contains('open')"):
        page.locator(f'.section[data-section="{group}"] .advanced-toggle').first.click()
    _settle(page, 500)


def _shoot_round6_evidence(page, base_url: str, out_dir: Path, illegal_id: str,
                           shots: list[str]) -> None:
    """Round-6: the Deployment-target panel in the LEFT column below Spiking
    semantics; the legal-value-set law (locked TTFS fields, legal-subset
    options); the CONCRETE green derived values with their provenance badges;
    and an illegal document surfacing as a keyed error with a one-click remedy."""
    # Item 1 — layout: Spiking semantics then Deployment target, same column.
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "semantics")
    _shot(page, out_dir, "round6_1_layout_deployment_target_left_column", shots, full=True)

    # Item 3 — |legal| > 1: the widget offers ONLY the legal options. Under lif
    # the rate encoder refuses 'TTFS', so that segment does not render at all.
    _open_advanced(page, "spiking")
    _shot_element(page.locator('.section[data-section="spiking"]'), out_dir,
                  "round6_3_legal_subset_options_lif", shots)

    # Item 2 + amendment — the Deployment-target knobs render the CONCRETE value
    # their SSOT deriver produces, in faded green, with the source badge.
    _open_advanced(page, "deployment_target")
    _shot_element(page.locator('.section[data-section="deployment_target"]'), out_dir,
                  "round6_derived_infield_green_values", shots)
    _shot(page, out_dir, "round6_2_provenance_badges_deployment_target", shots, full=True)

    # Item 3 — |legal| == 1: spiking_mode='ttfs' forces firing_mode='TTFS' and
    # spike_generation_mode='TTFS'; both fields LOCK, read-only, showing the
    # derived value. thresholding_mode keeps two legal values and stays editable.
    _click_segment(page, "spiking_mode", "ttfs")
    _settle(page, 1200)
    _open_advanced(page, "spiking")
    _shot_element(page.locator('.section[data-section="spiking"]'), out_dir,
                  "round6_3_locked_ttfs_fields", shots)
    _shot(page, out_dir, "round6_3_locked_ttfs_fields_full", shots, full=True)

    # The cascaded schedule moves the nf_scm sample count from 2 to 64: the
    # green value is re-derived per resolve, not a static string.
    _click_segment(page, "spiking_mode", "ttfs_cycle_based")
    _settle(page, 1200)
    _open_advanced(page, "deployment_target")
    _shot_element(page.locator('.section[data-section="deployment_target"]'), out_dir,
                  "round6_derived_infield_mode_aware_cascaded", shots)

    # Item 4 — an illegal document (spiking_mode=lif, firing_mode=TTFS) used to
    # raise ValueError out of DeploymentPlan.resolve. It is now a keyed error.
    page.goto(base_url + "/wizard?template_id=" + illegal_id)
    _settle(page, 2400)
    _goto_section(page, "semantics")
    _shot(page, out_dir, "round6_4_illegal_document_keyed_error", shots, full=True)
    _shot_element(page.locator('.field[data-key="firing_mode"]'), out_dir,
                  "round6_4_keyed_error_with_remedy", shots)
    page.click('.error-remedies .btn-sm:has-text("Clear Firing Mode")')
    _settle(page, 1600)
    _shot(page, out_dir, "round6_4_after_one_click_remedy", shots, full=True)


def _shoot_round6_visual_language(page, base_url: str, out_dir: Path,
                                  shots: list[str]) -> None:
    """Round-6 items 6+7: provenance occupies NO layout (it is tooltip text),
    and ONE green language marks derivation-owned values on EVERY widget type —
    inputs, toggles, slider thumbs + numeric boxes, segmented ghosts, locked
    fields — while a user-OWNED value reads the normal active theme."""
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "semantics")
    _open_advanced(page, "deployment_target")

    # Item 6: the source chips are gone from the layout; hovering the field
    # reveals the doc + bounds + what empty resolves to + the SSOT source.
    page.hover('.field[data-key="nf_scm_parity_samples"]')
    page.wait_for_timeout(400)
    _shot(page, out_dir, "round6_6_provenance_in_tooltip", shots)

    # Item 7: toggles and sliders carrying derived values read green, and the
    # numeric boxes state the concrete value.
    _shot_element(page.locator('.section[data-section="deployment_target"]'), out_dir,
                  "round6_7_derived_language_toggles_sliders", shots)

    # Owned vs derived on the SAME widget types. capacity_gate takes an explicit
    # OFF; scm_torch_sim_parity_check is clicked twice to an explicit ON (the
    # user's cyan, beside the derivation's green ON above it); one slider gets a
    # hand value through its numeric box.
    page.click('.field[data-key="capacity_gate"] .toggle-row')
    page.click('.field[data-key="scm_torch_sim_parity_check"] .toggle-row')
    page.click('.field[data-key="scm_torch_sim_parity_check"] .toggle-row')
    box = page.locator('.field[data-key="onchip_majority_min_fraction"] .slider-combo-box')
    box.fill("0.35")
    box.dispatch_event("change")
    _settle(page, 1400)
    _shot_element(page.locator('.section[data-section="deployment_target"]'), out_dir,
                  "round6_7_owned_vs_derived_contrast", shots)

    # Tuning sliders + Hardware capability toggles: the same language, no card
    # and no key opts out.
    page.goto(base_url + "/wizard")
    _settle(page, 2200)
    _goto_section(page, "training")
    _open_advanced(page, "tuning")
    _shot_element(page.locator('.section[data-section="tuning"]'), out_dir,
                  "round6_7_derived_language_tuning_sliders", shots)
    _goto_section(page, "codesign")
    _shot_element(page.locator('.section[data-section="hardware"]'), out_dir,
                  "round6_7_derived_language_hardware_toggles", shots)


def _shoot_mode_switches(page, base_url: str, out_dir: Path, shots: list[str]) -> None:
    """Starter + each single mode switch: semantics + rail stay green."""
    for mode, schedule in MODE_SWITCHES:
        page.goto(base_url + "/wizard")
        _settle(page, 2200)
        _goto_section(page, "semantics")
        _click_segment(page, "spiking_mode", mode)
        if schedule is not None:
            _click_segment(page, "ttfs_cycle_schedule", schedule)
        _settle(page, 900)
        name = mode if schedule is None else f"{mode}_{schedule}"
        _shot(page, out_dir, f"mode_{name}", shots, full=True)


def _shoot_error_flow(page, base_url: str, out_dir: Path, template_id: str,
                      shots: list[str]) -> None:
    page.goto(base_url + "/wizard?template_id=" + template_id)
    _settle(page, 2200)
    _goto_section(page, "review")
    _shot(page, out_dir, "error_1_contract_violation", shots, full=True)
    page.click('.error-remedies .btn-sm:text-is("Declare vanilla (float weights)")')
    _settle(page, 1400)
    _shot(page, out_dir, "error_2_after_one_click_remedy", shots, full=True)


def _save_template(base_url: str, name: str, config: dict) -> str:
    import urllib.request

    req = urllib.request.Request(
        base_url + "/api/templates",
        data=json.dumps({"name": name, "config": config}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read())["id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO_ROOT / "generated" / "_wizard_review" / "round6"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Templates land in a scratch dir so the repo templates/ stays untouched.
    templates_dir = tempfile.mkdtemp(prefix="wizard_review_templates_")
    os.environ["MIMARSINAN_TEMPLATES_DIR"] = templates_dir
    os.environ.setdefault("MIMARSINAN_GUI_NO_BROWSER", "1")
    os.chdir(REPO_ROOT)

    port = _free_port()
    _start_server(port)
    base_url = f"http://127.0.0.1:{port}"

    with open(TEMPLATE_TIER_CONFIG, encoding="utf-8") as f:
        tier_config = json.load(f)
    template_id = _save_template(base_url, tier_config["experiment_name"], tier_config)

    # A deliberate quantization-contract violation for the error/remedy flow.
    broken = json.loads(json.dumps(tier_config))
    broken["experiment_name"] = "wq_contract_demo"
    broken["pipeline_mode"] = "phased"
    broken["deployment_parameters"]["weight_quantization"] = False
    broken["platform_constraints"]["weight_bits"] = 5
    broken_id = _save_template(base_url, "wq_contract_demo", broken)

    # A deliberately ILLEGAL document for the legal-value-set error flow: the
    # rate-coded LIF family admits firing_mode in {Default, Novena}.
    illegal = json.loads(json.dumps(tier_config))
    illegal["experiment_name"] = "illegal_firing_mode_demo"
    illegal["deployment_parameters"]["spiking_mode"] = "lif"
    illegal["deployment_parameters"]["firing_mode"] = "TTFS"
    illegal_id = _save_template(base_url, "illegal_firing_mode_demo", illegal)

    from playwright.sync_api import sync_playwright

    shots: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.on("console", lambda msg: msg.type == "error" and print(f"[console.error] {msg.text}"))
        page.on("pageerror", lambda err: print(f"[pageerror] {err}"))
        _shoot_sections(page, base_url, out_dir, "fresh", shots)
        _shoot_codesign_sequence(page, base_url, out_dir, shots)
        _shoot_cosearch_and_placeholders(page, base_url, out_dir, shots)
        _shoot_defect_evidence(page, base_url, out_dir, shots)
        _shoot_round3_evidence(page, base_url, out_dir, shots)
        _shoot_round4_evidence(page, base_url, out_dir, broken_id, shots)
        _shoot_round5_evidence(page, base_url, out_dir, shots)
        _shoot_round6_evidence(page, base_url, out_dir, illegal_id, shots)
        _shoot_round6_visual_language(page, base_url, out_dir, shots)
        _shoot_mode_switches(page, base_url, out_dir, shots)
        _shoot_sections(page, base_url, out_dir, "template", shots,
                        query="?template_id=" + template_id)
        _shoot_error_flow(page, base_url, out_dir, broken_id, shots)
        browser.close()

    print("Saved:")
    for shot in shots:
        print(" ", shot)


if __name__ == "__main__":
    main()
