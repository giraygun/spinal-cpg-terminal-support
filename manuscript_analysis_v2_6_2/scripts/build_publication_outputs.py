#!/usr/bin/env python3
"""Build locked publication tables R1--R6 and Figures 4--8.

The script consumes only the final read-only A--H derived outputs.  All
summaries are descriptive design-grid compressions within one frozen network
realization.  It intentionally produces no sampling-inference quantities.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cpg_v2_6_2_matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader


SCRIPT_VERSION = "publication-outputs-1.0.0"
EPSILON = 1e-12
SCRIPT_PATH = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT_PATH.parent.parent
REPOSITORY_ROOT = ANALYSIS_DIR.parent
DERIVED_DIR = REPOSITORY_ROOT / "derived" / "manuscript_analysis_v2_6_2"
PUBLICATION_DIR = DERIVED_DIR / "publication_outputs"
TABLE_DIR = PUBLICATION_DIR / "tables"
FIGURE_DIR = PUBLICATION_DIR / "figures"
QA_DIR = PUBLICATION_DIR / "qa"
SPEC_PATH = ANALYSIS_DIR / "PROTOCOL_SPEC.json"
LOCKED_PROTOCOL_PATH = ANALYSIS_DIR / "ANALYSIS_PROTOCOL_LOCKED_2026-08-26.md"
MASTER_QC_PATH = DERIVED_DIR / "master_analysis_qc.json"

SPEED_ORDER = ["low", "medium", "high"]
LOAD_ORDER = ["normal", "unilateral", "bilateral_high"]
PULSE_ORDER = ["none", "excitatory", "inhibitory"]
POP_ORDER = ["RG", "PF", "MN", "V0D", "V0V", "V2a", "V3", "V1Ia", "V1Ren", "V2b"]
SINGLE_ORDER = ["V1Ia", "V1Ren", "V2b", "V2a", "V0D", "V0V", "V3", "Ia", "Ib", "groupI"]
PAIR_ORDER = ["V0D+V0V", "V1Ia+V2b", "V2a+V0V", "V1Ren+V2b", "V3+Ia", "V3+Ib"]
WINDOW_ORDER = ["baseline", "stress_prechallenge", "stress_challenge", "recovery"]
FAST_ORDER = ["dynamic", "static_mean", "yoked", "off"]
MT_ORDER = ["static_matched", "time_yoked", "spatial_shuffled", "impaired", "off"]

PULSE_LABEL = {"none": "No pulse", "excitatory": "Excitatory", "inhibitory": "Inhibitory"}
SPEED_LABEL = {"low": "Low", "medium": "Medium", "high": "High"}
LOAD_LABEL = {"normal": "Normal", "unilateral": "Unilateral", "bilateral_high": "Bilateral high"}
WINDOW_LABEL = {
    "baseline": "Baseline",
    "stress_prechallenge": "Stress, pre-challenge",
    "stress_challenge": "Stress + challenge",
    "recovery": "Recovery",
}
FAST_LABEL = {"dynamic": "Dynamic KCa", "static_mean": "Static-mean KCa", "yoked": "Yoked KCa", "off": "KCa off"}
MT_LABEL = {
    "static_matched": "Static-matched",
    "time_yoked": "Time-yoked",
    "spatial_shuffled": "Spatially shuffled",
    "impaired": "Impaired",
    "off": "Support off",
}

# Okabe--Ito-inspired palettes.
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#777777"
BLACK = "#222222"
DIVERGING = LinearSegmentedColormap.from_list("oi_diverging", [BLUE, "#F7F7F7", VERMILION])
SEQUENTIAL = LinearSegmentedColormap.from_list("oi_sequential", ["#F7FBFF", SKY, BLUE])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(name: str) -> pd.DataFrame:
    path = DERIVED_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def finite(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if pd.isna(value):
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def describe_grid(values: Iterable[Any], prefix: str) -> dict[str, Any]:
    usable = finite(values)
    if not usable:
        return {
            f"{prefix}_complete_cells": 0,
            f"{prefix}_median": np.nan,
            f"{prefix}_range_min": np.nan,
            f"{prefix}_range_max": np.nan,
            f"{prefix}_negative_cells": 0,
            f"{prefix}_neutral_cells": 0,
            f"{prefix}_positive_cells": 0,
        }
    arr = np.asarray(usable, dtype=float)
    return {
        f"{prefix}_complete_cells": int(arr.size),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_range_min": float(np.min(arr)),
        f"{prefix}_range_max": float(np.max(arr)),
        f"{prefix}_negative_cells": int(np.sum(arr < -EPSILON)),
        f"{prefix}_neutral_cells": int(np.sum(np.abs(arr) <= EPSILON)),
        f"{prefix}_positive_cells": int(np.sum(arr > EPSILON)),
    }


def value_for_endpoint(frame: pd.DataFrame, endpoint: str, value_field: str) -> float:
    selected = frame.loc[frame["endpoint"] == endpoint, value_field]
    usable = finite(selected)
    return np.nan if not usable else usable[0]


def metadata_columns(
    frame: pd.DataFrame,
    generated_at: str,
    script_hash: str,
    spec_hash: str,
    protocol_hash: str,
    analysis_hashes: dict[str, str],
) -> pd.DataFrame:
    out = frame.copy()
    out["aggregation_basis"] = out.get(
        "aggregation_basis",
        "descriptive design grid within one frozen realization; ranges are not uncertainty intervals",
    )
    out["independent_stochastic_realization_count"] = 1
    out["inferential_scope"] = "conditional on one frozen realization; no sampling inference"
    out["generated_at_utc"] = generated_at
    out["publication_script_version"] = SCRIPT_VERSION
    out["publication_script_sha256"] = script_hash
    out["protocol_spec_sha256"] = spec_hash
    out["locked_protocol_sha256"] = protocol_hash
    out["source_a_to_d_script_sha256"] = analysis_hashes["analyze_a_to_d.py"]
    out["source_e_f_script_sha256"] = analysis_hashes["analyze_e_f.py"]
    out["source_g_h_script_sha256"] = analysis_hashes["analyze_g_h.py"]
    out["output_row_count"] = len(out)
    return out


def save_table(
    filename: str,
    frame: pd.DataFrame,
    registry: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
) -> pd.DataFrame:
    enriched = metadata_columns(frame, **provenance)
    path = TABLE_DIR / filename
    enriched.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
    registry[filename] = {
        "rows": int(len(enriched)),
        "columns": int(len(enriched.columns)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    return enriched


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.8, colors=BLACK)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")


def heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    *,
    diverging: bool,
    vmin: float | None = None,
    vmax: float | None = None,
    annotation_format: str | None = None,
    colorbar_label: str = "",
    diagonal_outline: bool = False,
) -> mpl.collections.QuadMesh:
    data = np.ma.masked_invalid(np.asarray(matrix, dtype=float))
    if diverging:
        max_abs = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
        max_abs = max(max_abs, EPSILON)
        vmin = -max_abs if vmin is None else vmin
        vmax = max_abs if vmax is None else vmax
        cmap = DIVERGING.copy()
    else:
        vmin = float(np.nanmin(matrix)) if vmin is None and np.isfinite(matrix).any() else (0.0 if vmin is None else vmin)
        vmax = float(np.nanmax(matrix)) if vmax is None and np.isfinite(matrix).any() else (1.0 if vmax is None else vmax)
        if abs(vmax - vmin) <= EPSILON:
            vmax = vmin + 1.0
        cmap = SEQUENTIAL.copy()
    cmap.set_bad("#D9D9D9")
    mesh = ax.pcolormesh(
        np.arange(matrix.shape[1] + 1),
        np.arange(matrix.shape[0] + 1),
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        edgecolors="white",
        linewidth=0.55,
    )
    ax.set_xlim(0, matrix.shape[1])
    ax.set_ylim(matrix.shape[0], 0)
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5, labels=column_labels, rotation=40, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5, labels=row_labels)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if annotation_format:
        threshold = (abs(vmin) + abs(vmax)) * 0.28 if diverging else vmin + 0.55 * (vmax - vmin)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if not np.isfinite(value):
                    continue
                color = "white" if (abs(value) > threshold if diverging else value > threshold) else BLACK
                ax.text(j + 0.5, i + 0.5, format(value, annotation_format), ha="center", va="center", fontsize=7.2, color=color)
    if diagonal_outline:
        for index in range(min(matrix.shape)):
            ax.add_patch(Rectangle((index, index), 1, 1, fill=False, edgecolor=BLACK, linewidth=1.1))
    if colorbar_label:
        colorbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.045, pad=0.03)
        colorbar.set_label(colorbar_label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    return mesh


def save_figure(fig: plt.Figure, number: int, figure_registry: dict[str, dict[str, Any]]) -> None:
    base = f"Figure_{number}"
    png_path = FIGURE_DIR / f"{base}.png"
    pdf_path = FIGURE_DIR / f"{base}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(png_path) as image:
        dpi = image.info.get("dpi", (0, 0))
        png_info = {
            "width_px": int(image.width),
            "height_px": int(image.height),
            "dpi_x": float(dpi[0]),
            "dpi_y": float(dpi[1]),
            "mode": image.mode,
        }
    pdf_reader = PdfReader(str(pdf_path))
    figure_registry[base] = {
        "png": {
            **png_info,
            "bytes": png_path.stat().st_size,
            "sha256": sha256_file(png_path),
        },
        "pdf": {
            "pages": len(pdf_reader.pages),
            "bytes": pdf_path.stat().st_size,
            "sha256": sha256_file(pdf_path),
        },
    }


def endpoint_name_for_pulse(pulse: str, axis: str) -> str:
    return f"baseline_{axis}_phase_error_deg" if pulse == "none" else f"post_pulse_{axis}_phase_error_deg"


def make_r1(
    spec: dict[str, Any],
    a_inventory: pd.DataFrame,
    e_atomic: pd.DataFrame,
    f_atomic: pd.DataFrame,
    g_epoch: pd.DataFrame,
    h_conditions: pd.DataFrame,
) -> pd.DataFrame:
    roles = {
        "A": "Intact descending-drive/load grid and active pulse controls",
        "B": "Single population or afferent interventions",
        "C": "Paired interventions and nonadditivity",
        "D": "Descending-drive-dependent recruitment and effect modification",
        "E": "Presynaptic route impairments",
        "F": "Class-by-route 2x2 factorial cells",
        "G": "Long composite neural, synaptic, and mechanical stress",
        "H": "Terminal-support timing/location and KCa controls",
    }
    units = {
        "A": "Exact descending-drive-by-load cell; active pulse matched to direction-specific sham",
        "B": "Intervention minus exact context-matched intact A condition",
        "C": "Pair minus intact and exact four-arm nonadditivity",
        "D": "Normal-load descending-drive profile and exact high-minus-low difference-in-differences",
        "E": "Route impairment minus exact context-matched intact A condition",
        "F": "Task-defined A0M0/A1M0/A0M1/A1M1 cell; simulation reuse is not replication",
        "G": "Route-by-window exact challenge-by-impairment difference-in-differences",
        "H": "Exact dynamic-minus-control and KCa difference-in-differences",
    }
    a_index = a_inventory.set_index("stage")
    e_one = e_atomic.loc[e_atomic["endpoint"] == "lr_phase_error_mean_abs_deg"].copy()
    f_one = f_atomic.loc[f_atomic["endpoint"] == "lr_phase_error_mean_abs_deg"].copy()
    f_tasks: dict[str, dict[str, Any]] = {}
    for arm in ["a0m0", "a1m0", "a0m1", "a1m1"]:
        for _, cell in f_one.iterrows():
            task = str(cell[f"{arm}_task_id"])
            f_tasks[task] = {
                "simulation_id": str(cell[f"{arm}_simulation_id"]),
                "rhythmic_failure": int(cell[f"{arm}_rhythmic_failure"]),
                "pulse_required": int(cell[f"{arm}_pulse_required"]),
                "pulse_delivered": int(cell[f"{arm}_pulse_delivered"]),
            }
    stage_failures = {
        **{stage: int(a_index.loc[stage, "rhythmic_failure_task_count"]) for stage in ["A", "B", "C", "D"]},
        "E": int(e_one["intervention_rhythmic_failure"].sum()),
        "F": int(sum(record["rhythmic_failure"] for record in f_tasks.values())),
        "G": int(g_epoch.groupby("simulation_id")["rhythmic_failure"].max().sum()),
        "H": int(h_conditions["rhythmic_failure"].sum()),
    }
    stage_unique_failure_simulations = {
        **{stage: 0 for stage in ["A", "B", "C", "D"]},
        "E": int(e_one.loc[e_one["intervention_rhythmic_failure"] == 1, "intervention_simulation_id"].nunique()),
        "F": len({record["simulation_id"] for record in f_tasks.values() if record["rhythmic_failure"] == 1}),
        "G": int(g_epoch.loc[g_epoch["rhythmic_failure"] == 1, "simulation_id"].nunique()),
        "H": int(h_conditions.loc[h_conditions["rhythmic_failure"] == 1, "simulation_id"].nunique()),
    }
    pulse_required = {
        **{stage: int(a_index.loc[stage, "pulse_required_task_count"]) for stage in ["A", "B", "C", "D"]},
        "E": int(e_one["intervention_pulse_required"].sum()),
        "F": int(sum(record["pulse_required"] for record in f_tasks.values())),
        "G": 0,
        "H": int(h_conditions["pulse_required"].sum()),
    }
    pulse_delivered = {
        **{stage: int(a_index.loc[stage, "pulse_delivered_task_count"]) for stage in ["A", "B", "C", "D"]},
        "E": int(e_one["intervention_pulse_delivered"].sum()),
        "F": int(sum(record["pulse_delivered"] for record in f_tasks.values())),
        "G": 0,
        "H": int(h_conditions["pulse_delivered"].sum()),
    }
    rows = []
    for stage in list("ABCDEFGH"):
        rows.append(
            {
                "stage": stage,
                "scientific_role": roles[stage],
                "task_count": int(spec["expected_counts"]["stage_tasks"][stage]),
                "unique_simulation_count": int(spec["expected_counts"]["stage_unique_simulations"][stage]),
                "technical_invalid_count": 0,
                "scientific_invalid_count": 0,
                "rhythmic_failure_task_or_simulation_count": stage_failures[stage],
                "unique_rhythmic_failure_simulation_count": stage_unique_failure_simulations[stage],
                "pulse_required_task_count": pulse_required[stage],
                "pulse_delivered_task_count": pulse_delivered[stage],
                "paired_analysis_rule": units[stage],
            }
        )
    return pd.DataFrame(rows)


def make_r2(a_grid: pd.DataFrame, a_pulse: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered_grid = a_grid.copy()
    ordered_grid["_speed_order"] = ordered_grid["speed"].map({name: index for index, name in enumerate(SPEED_ORDER)})
    ordered_grid["_load_order"] = ordered_grid["load"].map({name: index for index, name in enumerate(LOAD_ORDER)})
    for _, row in ordered_grid.sort_values(["_speed_order", "_load_order"]).iterrows():
        rows.append(
            {
                "record_type": "intact_no_pulse_exact_cell",
                "speed": row["speed"],
                "load": row["load"],
                "pulse": "none",
                "design_cell_count": 1,
                "rhythmic_to_rhythmic_count": 1 - int(row["rhythmic_failure"]),
                "rhythmic_to_failure_count": 0,
                "lr_phase_error_deg": row["lr_phase_error_mean_abs_deg"],
                "fe_phase_error_deg": row["fe_phase_error_mean_abs_deg"],
                "lr_phase_slip_count": row["lr_phase_slip_count"],
                "lr_phase_cycle_count": row["lr_phase_cycle_count"],
                "lr_phase_slip_fraction": row["lr_phase_slip_rate"],
                "fe_phase_slip_count": row["fe_phase_slip_count"],
                "fe_phase_cycle_count": row["fe_phase_cycle_count"],
                "fe_phase_slip_fraction": row["fe_phase_slip_rate"],
                "frequency_hz": row["frequency_hz"],
                "rg_cycle_interval_cv": row["rg_cycle_interval_cv_mean"],
                "lr_pulse_minus_sham_median_deg": np.nan,
                "lr_pulse_minus_sham_range_min_deg": np.nan,
                "lr_pulse_minus_sham_range_max_deg": np.nan,
                "lr_pulse_minus_sham_positive_cells": np.nan,
                "lr_pulse_minus_sham_negative_cells": np.nan,
                "fe_pulse_minus_sham_median_deg": np.nan,
                "fe_pulse_minus_sham_range_min_deg": np.nan,
                "fe_pulse_minus_sham_range_max_deg": np.nan,
                "fe_pulse_minus_sham_positive_cells": np.nan,
                "fe_pulse_minus_sham_negative_cells": np.nan,
                "recovery_eligible_cells": np.nan,
                "recovery_event_observed_cells": np.nan,
                "recovery_eligible_without_event_cells": np.nan,
                "recovery_ineligible_cells": np.nan,
                "recovery_observed_time_median_s": np.nan,
                "recovery_observed_time_range_min_s": np.nan,
                "recovery_observed_time_range_max_s": np.nan,
            }
        )
    for pulse in ["excitatory", "inhibitory"]:
        part = a_pulse.loc[a_pulse["pulse"] == pulse]
        lr = describe_grid(part.loc[part["lr_phase_complete_pair"] == 1, "lr_phase_delta_deg"], "lr")
        fe = describe_grid(part.loc[part["fe_phase_complete_pair"] == 1, "fe_phase_delta_deg"], "fe")
        recovery_times = part.loc[part["recovery_event_observed"] == 1, "recovery_observed_time_s"]
        rec = describe_grid(recovery_times, "recovery")
        eligible = int(part["recovery_eligible"].sum())
        events = int(part["recovery_event_observed"].sum())
        rows.append(
            {
                "record_type": "active_pulse_minus_direction_matched_sham_summary",
                "speed": "all",
                "load": "all",
                "pulse": pulse,
                "design_cell_count": len(part),
                "rhythmic_to_rhythmic_count": int((part["failure_transition"] == "rhythmic_to_rhythmic").sum()),
                "rhythmic_to_failure_count": int((part["failure_transition"] == "rhythmic_to_failure").sum()),
                "lr_phase_error_deg": np.nan,
                "fe_phase_error_deg": np.nan,
                "lr_phase_slip_count": int(part["active_post_lr_phase_slip_count"].sum()),
                "lr_phase_cycle_count": int(part["active_post_lr_phase_cycle_count"].sum()),
                "lr_phase_slip_fraction": np.nan,
                "fe_phase_slip_count": int(part["active_post_fe_phase_slip_count"].sum()),
                "fe_phase_cycle_count": int(part["active_post_fe_phase_cycle_count"].sum()),
                "fe_phase_slip_fraction": np.nan,
                "frequency_hz": np.nan,
                "rg_cycle_interval_cv": np.nan,
                "lr_pulse_minus_sham_median_deg": lr["lr_median"],
                "lr_pulse_minus_sham_range_min_deg": lr["lr_range_min"],
                "lr_pulse_minus_sham_range_max_deg": lr["lr_range_max"],
                "lr_pulse_minus_sham_positive_cells": lr["lr_positive_cells"],
                "lr_pulse_minus_sham_negative_cells": lr["lr_negative_cells"],
                "fe_pulse_minus_sham_median_deg": fe["fe_median"],
                "fe_pulse_minus_sham_range_min_deg": fe["fe_range_min"],
                "fe_pulse_minus_sham_range_max_deg": fe["fe_range_max"],
                "fe_pulse_minus_sham_positive_cells": fe["fe_positive_cells"],
                "fe_pulse_minus_sham_negative_cells": fe["fe_negative_cells"],
                "recovery_eligible_cells": eligible,
                "recovery_event_observed_cells": events,
                "recovery_eligible_without_event_cells": eligible - events,
                "recovery_ineligible_cells": len(part) - eligible,
                "recovery_observed_time_median_s": rec["recovery_median"],
                "recovery_observed_time_range_min_s": rec["recovery_range_min"],
                "recovery_observed_time_range_max_s": rec["recovery_range_max"],
            }
        )
    return pd.DataFrame(rows)


def summarize_b_or_c(
    continuous: pd.DataFrame,
    states: pd.DataFrame,
    group_field: str,
    comparison_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = continuous[[group_field, "pulse"]].drop_duplicates()
    for _, key in groups.iterrows():
        name, pulse = key[group_field], key["pulse"]
        part = continuous.loc[(continuous[group_field] == name) & (continuous["pulse"] == pulse)]
        state = states.loc[(states[group_field] == name) & (states["pulse"] == pulse)]
        lr_endpoint = endpoint_name_for_pulse(pulse, "lr")
        fe_endpoint = endpoint_name_for_pulse(pulse, "fe")
        if comparison_type == "single_minus_intact":
            value_field = "delta_intervention_minus_reference"
            complete_field = "complete_pair"
            failure_field = "failure_transition"
            recovery_eligible_field = "intervention_recovery_eligible"
            recovery_event_field = "intervention_recovery_event"
            recovery_time_field = "intervention_recovery_observed_time_s"
        else:
            value_field = "pair_minus_intact"
            complete_field = "pair_vs_intact_complete_pair"
            failure_field = "pair_vs_intact_failure_transition"
            recovery_eligible_field = "pair_recovery_eligible"
            recovery_event_field = "pair_recovery_event"
            recovery_time_field = "pair_recovery_observed_time_s"
        lr_values = part.loc[(part["endpoint"] == lr_endpoint) & (part[complete_field] == 1), value_field]
        fe_values = part.loc[(part["endpoint"] == fe_endpoint) & (part[complete_field] == 1), value_field]
        row: dict[str, Any] = {
            "comparison_type": comparison_type,
            "intervention_or_pair": name,
            "pulse": pulse,
            "phase_analysis_window": "whole_run" if pulse == "none" else "post_pulse",
            "design_cell_count": len(state),
            "rhythmic_to_rhythmic_count": int((state[failure_field] == "rhythmic_to_rhythmic").sum()),
            "rhythmic_to_failure_count": int((state[failure_field] == "rhythmic_to_failure").sum()),
            "failure_to_rhythmic_count": int((state[failure_field] == "failure_to_rhythmic").sum()),
            "failure_to_failure_count": int((state[failure_field] == "failure_to_failure").sum()),
            **describe_grid(lr_values, "lr_phase_delta_deg"),
            **describe_grid(fe_values, "fe_phase_delta_deg"),
        }
        if comparison_type == "pair_minus_intact":
            lr_nonadd = part.loc[(part["endpoint"] == lr_endpoint) & (part["complete_quad"] == 1), "nonadditivity_pair_minus_singles_plus_intact"]
            fe_nonadd = part.loc[(part["endpoint"] == fe_endpoint) & (part["complete_quad"] == 1), "nonadditivity_pair_minus_singles_plus_intact"]
            row.update(describe_grid(lr_nonadd, "lr_nonadditivity_deg"))
            row.update(describe_grid(fe_nonadd, "fe_nonadditivity_deg"))
        else:
            row.update(describe_grid([], "lr_nonadditivity_deg"))
            row.update(describe_grid([], "fe_nonadditivity_deg"))
        if pulse == "none":
            row.update(
                {
                    "recovery_eligible_cells": np.nan,
                    "recovery_event_observed_cells": np.nan,
                    "recovery_eligible_without_event_cells": np.nan,
                    "recovery_ineligible_cells": np.nan,
                    "recovery_observed_time_complete_cells": 0,
                    "recovery_observed_time_median_s": np.nan,
                    "recovery_observed_time_range_min_s": np.nan,
                    "recovery_observed_time_range_max_s": np.nan,
                }
            )
        else:
            eligible = int(state[recovery_eligible_field].sum())
            events = int(state[recovery_event_field].sum())
            times = state.loc[state[recovery_event_field] == 1, recovery_time_field]
            desc = describe_grid(times, "recovery_observed_time")
            row.update(
                {
                    "recovery_eligible_cells": eligible,
                    "recovery_event_observed_cells": events,
                    "recovery_eligible_without_event_cells": eligible - events,
                    "recovery_ineligible_cells": len(state) - eligible,
                    "recovery_observed_time_complete_cells": desc["recovery_observed_time_complete_cells"],
                    "recovery_observed_time_median_s": desc["recovery_observed_time_median"],
                    "recovery_observed_time_range_min_s": desc["recovery_observed_time_range_min"],
                    "recovery_observed_time_range_max_s": desc["recovery_observed_time_range_max"],
                }
            )
        rows.append(row)
    return rows


def make_r3(b_cont: pd.DataFrame, b_states: pd.DataFrame, c_cont: pd.DataFrame, c_states: pd.DataFrame) -> pd.DataFrame:
    rows = summarize_b_or_c(b_cont, b_states, "intervention", "single_minus_intact")
    rows += summarize_b_or_c(c_cont, c_states, "pair", "pair_minus_intact")
    frame = pd.DataFrame(rows)
    order = {name: index for index, name in enumerate(SINGLE_ORDER + PAIR_ORDER)}
    pulse_order = {name: index for index, name in enumerate(PULSE_ORDER)}
    frame["_order"] = frame["intervention_or_pair"].map(order)
    frame["_pulse"] = frame["pulse"].map(pulse_order)
    frame["_comparison"] = frame["comparison_type"].map({"single_minus_intact": 0, "pair_minus_intact": 1})
    return frame.sort_values(["_comparison", "_order", "_pulse"]).drop(columns=["_order", "_pulse", "_comparison"]).reset_index(drop=True)


def make_r4(e_phase: pd.DataFrame, e_prop: pd.DataFrame, e_terminal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = e_phase.loc[e_phase["endpoint"] == "lr_phase_error_mean_abs_deg"]
    for route in POP_ORDER:
        for pulse in PULSE_ORDER:
            state = base.loc[(base["intended_route"] == route) & (base["pulse"] == pulse)]
            row: dict[str, Any] = {
                "intended_route": route,
                "pulse": pulse,
                "design_cell_count": len(state),
                "rhythmic_to_rhythmic_count": int((state["failure_transition"] == "rhythmic_to_rhythmic").sum()),
                "rhythmic_to_failure_count": int((state["failure_transition"] == "rhythmic_to_failure").sum()),
                "failure_to_rhythmic_count": int((state["failure_transition"] == "failure_to_rhythmic").sum()),
                "failure_to_failure_count": int((state["failure_transition"] == "failure_to_failure").sum()),
            }
            for endpoint, prefix in [
                ("lr_phase_error_mean_abs_deg", "lr_phase_delta_deg"),
                ("fe_phase_error_mean_abs_deg", "fe_phase_delta_deg"),
            ]:
                part = e_phase.loc[
                    (e_phase["intended_route"] == route)
                    & (e_phase["pulse"] == pulse)
                    & (e_phase["endpoint"] == endpoint)
                    & (e_phase["complete_pair"] == 1),
                    "paired_delta_intervention_minus_reference",
                ]
                row.update(describe_grid(part, prefix))
            for endpoint, prefix in [
                ("pf_missed_transfer_rate", "pf_propagation_gap_delta"),
                ("mn_missed_transfer_rate", "mn_propagation_gap_delta"),
            ]:
                part = e_prop.loc[
                    (e_prop["intended_route"] == route)
                    & (e_prop["pulse"] == pulse)
                    & (e_prop["endpoint"] == endpoint)
                    & (e_prop["complete_pair"] == 1),
                    "paired_delta_intervention_minus_reference",
                ]
                row.update(describe_grid(part, prefix))
            for endpoint, prefix in [
                ("intended_route_mt_support", "route_mt_support_delta"),
                ("intended_route_rrp", "route_rrp_delta"),
                ("intended_route_replenishment_resource", "route_replenishment_resource_delta"),
            ]:
                part = e_terminal.loc[
                    (e_terminal["intended_route"] == route)
                    & (e_terminal["pulse"] == pulse)
                    & (e_terminal["endpoint"] == endpoint)
                    & (e_terminal["complete_pair"] == 1),
                    "paired_delta_intervention_minus_reference",
                ]
                row.update(describe_grid(part, prefix))
            if pulse == "none":
                row.update(
                    {
                        "recovery_eligible_cells": np.nan,
                        "recovery_event_observed_cells": np.nan,
                        "recovery_eligible_without_event_cells": np.nan,
                        "recovery_ineligible_cells": np.nan,
                        **describe_grid([], "recovery_observed_time_s"),
                    }
                )
            else:
                eligible = int(state["intervention_recovery_endpoint_eligible"].sum())
                events = int(state["intervention_recovery_event_observed"].sum())
                times = state.loc[state["intervention_recovery_event_observed"] == 1, "intervention_recovery_time_s"]
                row.update(
                    {
                        "recovery_eligible_cells": eligible,
                        "recovery_event_observed_cells": events,
                        "recovery_eligible_without_event_cells": eligible - events,
                        "recovery_ineligible_cells": len(state) - eligible,
                        **describe_grid(times, "recovery_observed_time_s"),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def make_r5(f_phase: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    state_endpoint = "lr_phase_error_mean_abs_deg"
    for intended_class in POP_ORDER:
        for intended_route in POP_ORDER:
            for pulse in PULSE_ORDER:
                key = (
                    (f_phase["intended_class"] == intended_class)
                    & (f_phase["intended_route"] == intended_route)
                    & (f_phase["pulse"] == pulse)
                )
                state = f_phase.loc[key & (f_phase["endpoint"] == state_endpoint)]
                row: dict[str, Any] = {
                    "intended_class": intended_class,
                    "intended_route": intended_route,
                    "matrix_position": "diagonal" if intended_class == intended_route else "off_diagonal",
                    "pulse": pulse,
                    "design_cell_count": len(state),
                    "all_four_arms_rhythmic_cells": int(state["all_four_arms_rhythmic"].sum()),
                    "a1m0_rhythmic_failure_cells": int(state["a1m0_rhythmic_failure"].sum()),
                    "a1m1_rhythmic_failure_cells": int(state["a1m1_rhythmic_failure"].sum()),
                }
                for endpoint, axis in [
                    ("lr_phase_error_mean_abs_deg", "lr"),
                    ("fe_phase_error_mean_abs_deg", "fe"),
                ]:
                    part = f_phase.loc[key & (f_phase["endpoint"] == endpoint)]
                    class_effect = part["a1m0_value"] - part["a0m0_value"]
                    route_effect = part["a0m1_value"] - part["a0m0_value"]
                    joint_effect = part["a1m1_value"] - part["a0m0_value"]
                    nonadd = part.loc[part["complete_four_arm_cell"] == 1, "nonadditivity_a1m1_minus_a1m0_minus_a0m1_plus_a0m0"]
                    row.update(describe_grid(class_effect, f"{axis}_class_marginal_delta_deg"))
                    row.update(describe_grid(route_effect, f"{axis}_route_marginal_delta_deg"))
                    row.update(describe_grid(joint_effect, f"{axis}_joint_delta_deg"))
                    row.update(describe_grid(nonadd, f"{axis}_nonadditivity_deg"))
                if pulse == "none":
                    for arm in ["a0m0", "a1m0", "a0m1", "a1m1"]:
                        row[f"{arm}_recovery_eligible_cells"] = np.nan
                        row[f"{arm}_recovery_event_observed_cells"] = np.nan
                else:
                    for arm in ["a0m0", "a1m0", "a0m1", "a1m1"]:
                        row[f"{arm}_recovery_eligible_cells"] = int(state[f"{arm}_recovery_endpoint_eligible"].sum())
                        row[f"{arm}_recovery_event_observed_cells"] = int(state[f"{arm}_recovery_event_observed"].sum())
                rows.append(row)
    return pd.DataFrame(rows)


def make_r6(g_did: pd.DataFrame, h_mt: pd.DataFrame, h_fast: pd.DataFrame, h_did: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for route in POP_ORDER:
        for window in WINDOW_ORDER:
            part = g_did.loc[(g_did["route"] == route) & (g_did["window"] == window)]
            row: dict[str, Any] = {
                "analysis_block": "G_challenge_by_impairment",
                "route_or_mt_comparator": route,
                "window_or_fast_mode": window,
                "pulse": "not_applicable",
                "fast_comparator": "not_applicable",
                "formula": "(challenge impaired - no-challenge impaired) - (challenge intact - no-challenge intact)",
                "all_arms_status_valid": int(part["all_arms_status_valid"].all()),
                "all_arms_rhythmic": int(part["all_arms_rhythmic"].all()),
            }
            for endpoint, output in [
                ("lr_phase_error_mean_abs_deg", "lr_phase_effect_deg"),
                ("fe_phase_error_mean_abs_deg", "fe_phase_effect_deg"),
                ("pf_network_propagation_gap", "pf_propagation_gap_effect"),
                ("mn_network_propagation_gap", "mn_propagation_gap_effect"),
                ("route_mt_mean", "route_mt_effect"),
                ("route_rrp_mean_secondary", "route_rrp_effect"),
                ("route_replenishment_resource_mean_secondary", "route_replenishment_resource_effect"),
                ("rhythmic_failure_epoch_fraction", "failure_epoch_fraction_effect"),
            ]:
                row[output] = value_for_endpoint(part, endpoint, "contrast_value")
            rows.append(row)
    for fast_mode in FAST_ORDER:
        for pulse in PULSE_ORDER:
            for comparator in MT_ORDER:
                part = h_mt.loc[
                    (h_mt["fast_mode"] == fast_mode)
                    & (h_mt["pulse"] == pulse)
                    & (h_mt["mt_comparator"] == comparator)
                ]
                status = part.iloc[0]
                rows.append(
                    {
                        "analysis_block": "H_dynamic_minus_terminal_support_control",
                        "route_or_mt_comparator": comparator,
                        "window_or_fast_mode": fast_mode,
                        "pulse": pulse,
                        "fast_comparator": "not_applicable",
                        "formula": "dynamic terminal support - comparator terminal support",
                        "all_arms_status_valid": 1,
                        "all_arms_rhythmic": int(status["failure_transition_reference_to_target"] == "rhythmic_to_rhythmic"),
                        "lr_phase_effect_deg": value_for_endpoint(part, "analysis_lr_phase_error_mean_abs_deg", "contrast_value"),
                        "fe_phase_effect_deg": value_for_endpoint(part, "analysis_fe_phase_error_mean_abs_deg", "contrast_value"),
                        "pf_propagation_gap_effect": value_for_endpoint(part, "pf_network_propagation_gap", "contrast_value"),
                        "mn_propagation_gap_effect": value_for_endpoint(part, "mn_network_propagation_gap", "contrast_value"),
                        "route_mt_effect": np.nan,
                        "route_rrp_effect": np.nan,
                        "route_replenishment_resource_effect": np.nan,
                        "failure_epoch_fraction_effect": np.nan,
                        "recovery_reference_eligible": status["recovery_eligible_reference"],
                        "recovery_target_eligible": status["recovery_eligible_target"],
                        "recovery_reference_event_observed": status["recovery_event_reference"],
                        "recovery_target_event_observed": status["recovery_event_target"],
                        "recovery_observed_time_effect_s": value_for_endpoint(part, "recovery_time_s", "contrast_value"),
                    }
                )
    for mt_mode in ["dynamic", "static_matched", "time_yoked", "spatial_shuffled", "impaired", "off"]:
        for pulse in PULSE_ORDER:
            for fast_comparator in ["static_mean", "yoked", "off"]:
                part = h_fast.loc[
                    (h_fast["mt_mode"] == mt_mode)
                    & (h_fast["pulse"] == pulse)
                    & (h_fast["fast_comparator"] == fast_comparator)
                ]
                status = part.iloc[0]
                rows.append(
                    {
                        "analysis_block": "H_KCa_control_minus_dynamic_KCa",
                        "route_or_mt_comparator": mt_mode,
                        "window_or_fast_mode": "KCa_control",
                        "pulse": pulse,
                        "fast_comparator": fast_comparator,
                        "formula": "comparator KCa - dynamic KCa within the same terminal-support mode",
                        "all_arms_status_valid": 1,
                        "all_arms_rhythmic": int(status["failure_transition_reference_to_target"] == "rhythmic_to_rhythmic"),
                        "lr_phase_effect_deg": value_for_endpoint(part, "analysis_lr_phase_error_mean_abs_deg", "contrast_value"),
                        "fe_phase_effect_deg": value_for_endpoint(part, "analysis_fe_phase_error_mean_abs_deg", "contrast_value"),
                        "pf_propagation_gap_effect": value_for_endpoint(part, "pf_network_propagation_gap", "contrast_value"),
                        "mn_propagation_gap_effect": value_for_endpoint(part, "mn_network_propagation_gap", "contrast_value"),
                        "route_mt_effect": np.nan,
                        "route_rrp_effect": np.nan,
                        "route_replenishment_resource_effect": np.nan,
                        "failure_epoch_fraction_effect": np.nan,
                        "recovery_reference_eligible": status["recovery_eligible_reference"],
                        "recovery_target_eligible": status["recovery_eligible_target"],
                        "recovery_reference_event_observed": status["recovery_event_reference"],
                        "recovery_target_event_observed": status["recovery_event_target"],
                        "recovery_observed_time_effect_s": value_for_endpoint(part, "recovery_time_s", "contrast_value"),
                    }
                )
    for fast_comparator in ["static_mean", "yoked", "off"]:
        for pulse in PULSE_ORDER:
            for comparator in MT_ORDER:
                part = h_did.loc[
                    (h_did["fast_comparator"] == fast_comparator)
                    & (h_did["pulse"] == pulse)
                    & (h_did["mt_comparator"] == comparator)
                ]
                status = part.iloc[0]
                rows.append(
                    {
                        "analysis_block": "H_terminal_support_contrast_by_KCa_mode",
                        "route_or_mt_comparator": comparator,
                        "window_or_fast_mode": "KCa_difference_in_differences",
                        "pulse": pulse,
                        "fast_comparator": fast_comparator,
                        "formula": "MT contrast at comparator KCa - MT contrast at dynamic KCa",
                        "all_arms_status_valid": 1,
                        "all_arms_rhythmic": int(
                            status["failure_transition_dynamic_fast_mt_comparator_to_dynamic"] == "rhythmic_to_rhythmic"
                            and status["failure_transition_fast_mode_mt_comparator_to_dynamic"] == "rhythmic_to_rhythmic"
                        ),
                        "lr_phase_effect_deg": value_for_endpoint(part, "analysis_lr_phase_error_mean_abs_deg", "exact_difference_in_differences"),
                        "fe_phase_effect_deg": value_for_endpoint(part, "analysis_fe_phase_error_mean_abs_deg", "exact_difference_in_differences"),
                        "pf_propagation_gap_effect": value_for_endpoint(part, "pf_network_propagation_gap", "exact_difference_in_differences"),
                        "mn_propagation_gap_effect": value_for_endpoint(part, "mn_network_propagation_gap", "exact_difference_in_differences"),
                        "route_mt_effect": np.nan,
                        "route_rrp_effect": np.nan,
                        "route_replenishment_resource_effect": np.nan,
                        "failure_epoch_fraction_effect": np.nan,
                        "recovery_reference_eligible": np.nan,
                        "recovery_target_eligible": np.nan,
                        "recovery_reference_event_observed": np.nan,
                        "recovery_target_event_observed": np.nan,
                        "recovery_observed_time_effect_s": value_for_endpoint(part, "recovery_time_s", "exact_difference_in_differences"),
                    }
                )
    return pd.DataFrame(rows)


def matrix_from_long(frame: pd.DataFrame, row_field: str, column_field: str, value_field: str, rows: list[str], columns: list[str]) -> np.ndarray:
    pivot = frame.pivot_table(index=row_field, columns=column_field, values=value_field, aggfunc="first")
    return pivot.reindex(index=rows, columns=columns).to_numpy(dtype=float)


def build_figures(
    data: dict[str, pd.DataFrame],
    table_registry: dict[str, dict[str, Any]],
    figure_registry: dict[str, dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    # Figure 4: intact phase stability across the complete 3x3 design grid.
    a_grid = data["a_grid"].copy()
    panel_specs = [
        ("A", "lr_phase_error_mean_abs_deg", "L-R absolute phase error", "deg", True),
        ("B", "fe_phase_error_mean_abs_deg", "F-E absolute phase error", "deg", True),
        ("C", "lr_phase_slip_rate", "L-R phase-slip fraction", "fraction", False),
        ("D", "fe_phase_slip_rate", "F-E phase-slip fraction", "fraction", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.6), constrained_layout=True)
    for ax, (label, field, title, unit, annotate) in zip(axes.flat, panel_specs):
        panel = a_grid[["task_id", "simulation_id", "speed", "load", "rhythmic_failure", field]].rename(columns={field: "value"})
        save_table(f"Figure_4{label}_data.csv", panel, table_registry, provenance)
        matrix = matrix_from_long(panel, "speed", "load", "value", SPEED_ORDER, LOAD_ORDER)
        heatmap(
            ax,
            matrix,
            [SPEED_LABEL[x] for x in SPEED_ORDER],
            [LOAD_LABEL[x] for x in LOAD_ORDER],
            title,
            diverging=False,
            vmin=0,
            vmax=45 if unit == "deg" else max(0.1, float(np.nanmax(matrix))),
            annotation_format=".1f" if unit == "deg" else ".3f",
            colorbar_label=f"{unit}; 45° defines a slip" if unit == "deg" else unit,
        )
        ax.set_xlabel("Mechanical load")
        ax.set_ylabel("Descending-drive command")
        panel_label(ax, label)
    fig.suptitle("Intact-network phase stability across descending drive and load", fontsize=14, fontweight="bold")
    save_figure(fig, 4, figure_registry)

    # Figure 5: pulse-minus-sham phase effects and full-follow-up recovery.
    pulse = data["a_pulse"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.15, hspace=0.36, wspace=0.28)
    marker_by_speed = {"low": "o", "medium": "s", "high": "^"}
    color_by_load = {"normal": BLUE, "unilateral": ORANGE, "bilateral_high": GREEN}
    for ax, axis_name, label in [(axes[0, 0], "lr", "A"), (axes[0, 1], "fe", "B")]:
        field = f"{axis_name}_phase_delta_deg"
        panel = pulse[["active_task_id", "sham_reference_task_id", "speed", "load", "pulse", f"{axis_name}_phase_complete_pair", field]].copy()
        save_table(f"Figure_5{label}_data.csv", panel, table_registry, provenance)
        for pulse_index, pulse_name in enumerate(["excitatory", "inhibitory"]):
            subset = panel.loc[panel["pulse"] == pulse_name]
            offsets = {"normal": -0.16, "unilateral": 0.0, "bilateral_high": 0.16}
            for _, row in subset.iterrows():
                ax.scatter(
                    pulse_index + offsets[row["load"]],
                    row[field],
                    marker=marker_by_speed[row["speed"]],
                    color=color_by_load[row["load"]],
                    edgecolor="white",
                    linewidth=0.55,
                    s=58,
                    zorder=3,
                )
            median_value = float(subset[field].median())
            ax.plot([pulse_index - 0.25, pulse_index + 0.25], [median_value, median_value], color=BLACK, lw=2.0)
        ax.axhline(0, color=GRAY, linewidth=0.9, linestyle="--")
        ax.set_xticks([0, 1], ["Excitatory", "Inhibitory"])
        ax.set_ylabel(f"{axis_name.upper()} pulse - matched sham (deg)")
        ax.set_title(f"{axis_name.upper()} post-pulse phase effect")
        style_axes(ax)
        panel_label(ax, label)
    status_rows = []
    for pulse_name in ["excitatory", "inhibitory"]:
        subset = pulse.loc[pulse["pulse"] == pulse_name]
        eligible = int(subset["recovery_eligible"].sum())
        event = int(subset["recovery_event_observed"].sum())
        status_rows.extend(
            [
                {"pulse": pulse_name, "recovery_status": "Observed event", "design_cell_count": event},
                {"pulse": pulse_name, "recovery_status": "Eligible, no event", "design_cell_count": eligible - event},
                {"pulse": pulse_name, "recovery_status": "Ineligible", "design_cell_count": len(subset) - eligible},
            ]
        )
    status = pd.DataFrame(status_rows)
    save_table("Figure_5C_data.csv", status, table_registry, provenance)
    ax = axes[1, 0]
    bottom = np.zeros(2)
    status_colors = {"Observed event": GREEN, "Eligible, no event": ORANGE, "Ineligible": GRAY}
    for status_name in ["Observed event", "Eligible, no event", "Ineligible"]:
        values = [int(status.loc[(status["pulse"] == p) & (status["recovery_status"] == status_name), "design_cell_count"].iloc[0]) for p in ["excitatory", "inhibitory"]]
        ax.bar([0, 1], values, bottom=bottom, color=status_colors[status_name], edgecolor="white", label=status_name, width=0.65)
        bottom += np.asarray(values)
    ax.set_xticks([0, 1], ["Excitatory", "Inhibitory"])
    ax.set_ylabel("Drive-by-load cells")
    ax.set_ylim(0, 10)
    ax.set_title("Full-follow-up recovery status")
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    style_axes(ax)
    panel_label(ax, "C")
    recovery = pulse.loc[pulse["recovery_event_observed"] == 1, [
        "active_task_id", "speed", "load", "pulse", "recovery_eligible", "recovery_event_observed",
        "recovery_observed_time_s", "recovery_time_or_censor_s", "recovery_censor_time_s",
    ]].copy()
    save_table("Figure_5D_data.csv", recovery, table_registry, provenance)
    ax = axes[1, 1]
    for pulse_index, pulse_name in enumerate(["excitatory", "inhibitory"]):
        subset = recovery.loc[recovery["pulse"] == pulse_name]
        offsets = {"normal": -0.16, "unilateral": 0.0, "bilateral_high": 0.16}
        for _, row in subset.iterrows():
            ax.scatter(
                pulse_index + offsets[row["load"]], row["recovery_observed_time_s"],
                marker=marker_by_speed[row["speed"]], color=color_by_load[row["load"]],
                edgecolor="white", linewidth=0.55, s=58,
            )
    ax.set_xticks([0, 1], ["Excitatory", "Inhibitory"])
    ax.set_ylabel("Observed recovery time (s)")
    ax.set_title("Recovery times among observed events")
    style_axes(ax)
    panel_label(ax, "D")
    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", color=color_by_load[key], label=LOAD_LABEL[key], markersize=6)
        for key in LOAD_ORDER
    ] + [
        mpl.lines.Line2D([], [], marker=marker_by_speed[key], linestyle="", color=BLACK, label=f"{SPEED_LABEL[key]} drive", markersize=6)
        for key in SPEED_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("Pulse-evoked phase change and recovery in the intact network", fontsize=14, fontweight="bold", y=0.97)
    save_figure(fig, 5, figure_registry)

    # Figure 6: single, paired, nonadditive, and descending-drive recruitment views.
    r3 = data["r3"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0), constrained_layout=True)
    column_keys = [(p, a) for p in PULSE_ORDER for a in ["lr", "fe"]]
    column_labels = [f"{PULSE_LABEL[p]}\n{a.upper()}" for p, a in column_keys]
    for ax, comparison, row_order, value_kind, label, title in [
        (axes[0, 0], "single_minus_intact", SINGLE_ORDER, "phase_delta", "A", "Single intervention - intact"),
        (axes[0, 1], "pair_minus_intact", PAIR_ORDER, "phase_delta", "B", "Paired intervention - intact"),
        (axes[1, 0], "pair_minus_intact", PAIR_ORDER, "nonadditivity", "C", "Paired-intervention nonadditivity"),
    ]:
        part = r3.loc[r3["comparison_type"] == comparison].copy()
        panel_rows = []
        matrix = np.full((len(row_order), len(column_keys)), np.nan)
        for i, name in enumerate(row_order):
            for j, (pulse_name, axis_name) in enumerate(column_keys):
                match = part.loc[(part["intervention_or_pair"] == name) & (part["pulse"] == pulse_name)]
                field = f"{axis_name}_{value_kind}_deg_median"
                if field not in match.columns:
                    field = f"{axis_name}_{value_kind}_deg_median"
                value = match.iloc[0][field] if len(match) else np.nan
                matrix[i, j] = value
                panel_rows.append(
                    {
                        "intervention_or_pair": name,
                        "pulse": pulse_name,
                        "phase_axis": axis_name.upper(),
                        "summary_measure": value_kind,
                        "design_grid_median_deg": value,
                        "design_grid_range_min_deg": match.iloc[0].get(f"{axis_name}_{value_kind}_deg_range_min", np.nan) if len(match) else np.nan,
                        "design_grid_range_max_deg": match.iloc[0].get(f"{axis_name}_{value_kind}_deg_range_max", np.nan) if len(match) else np.nan,
                        "complete_cells": match.iloc[0].get(f"{axis_name}_{value_kind}_deg_complete_cells", 0) if len(match) else 0,
                    }
                )
        save_table(f"Figure_6{label}_data.csv", pd.DataFrame(panel_rows), table_registry, provenance)
        heatmap(ax, matrix, row_order, column_labels, title, diverging=True, colorbar_label="Median exact effect (deg)")
        panel_label(ax, label)
    recruitment = data["recruitment"].loc[data["recruitment"]["arm"] == "none"].copy()
    recruitment["pulse_speed"] = recruitment["pulse"].map(PULSE_LABEL) + "\n" + recruitment["speed"].map(SPEED_LABEL)
    recruitment_columns = [f"{PULSE_LABEL[p]}\n{SPEED_LABEL[s]}" for p in PULSE_ORDER for s in SPEED_ORDER]
    panel = recruitment[["task_id", "simulation_id", "pulse", "speed", "model_population", "mean_rate_hz", "rhythmic_failure"]].copy()
    save_table("Figure_6D_data.csv", panel, table_registry, provenance)
    matrix = matrix_from_long(recruitment, "model_population", "pulse_speed", "mean_rate_hz", POP_ORDER, recruitment_columns)
    heatmap(axes[1, 1], matrix, POP_ORDER, recruitment_columns, "Intact population recruitment", diverging=False, vmin=0, colorbar_label="Mean firing rate (Hz)")
    panel_label(axes[1, 1], "D")
    fig.suptitle("Circuit interventions and recruitment across descending-drive commands", fontsize=14, fontweight="bold")
    save_figure(fig, 6, figure_registry)

    # Figure 7: route engagement and full class-by-route matrices.
    r4 = data["r4"].copy()
    r5 = data["r5"].copy()
    fig = plt.figure(figsize=(18.0, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.45, 1, 1, 1])
    axis_a = fig.add_subplot(grid[0, 0])
    axis_b = fig.add_subplot(grid[1, 0])
    phase_columns = [(p, a) for p in PULSE_ORDER for a in ["lr", "fe"]]
    phase_labels = [f"{PULSE_LABEL[p]}\n{a.upper()}" for p, a in phase_columns]
    phase_matrix = np.full((10, 6), np.nan)
    phase_rows = []
    for i, route in enumerate(POP_ORDER):
        for j, (pulse_name, axis_name) in enumerate(phase_columns):
            match = r4.loc[(r4["intended_route"] == route) & (r4["pulse"] == pulse_name)].iloc[0]
            field = f"{axis_name}_phase_delta_deg_median"
            phase_matrix[i, j] = match[field]
            phase_rows.append(
                {
                    "intended_route": route,
                    "pulse": pulse_name,
                    "phase_axis": axis_name.upper(),
                    "design_grid_median_delta_deg": match[field],
                    "design_grid_range_min_deg": match[f"{axis_name}_phase_delta_deg_range_min"],
                    "design_grid_range_max_deg": match[f"{axis_name}_phase_delta_deg_range_max"],
                    "complete_cells": match[f"{axis_name}_phase_delta_deg_complete_cells"],
                }
            )
    save_table("Figure_7A_data.csv", pd.DataFrame(phase_rows), table_registry, provenance)
    heatmap(axis_a, phase_matrix, POP_ORDER, phase_labels, "Route impairment - intact", diverging=True, colorbar_label="Median phase effect (deg)")
    panel_label(axis_a, "A")
    terminal_columns = [(p, e) for p in PULSE_ORDER for e in ["mt", "rrp", "resource"]]
    terminal_pulse_label = {"none": "None", "excitatory": "Exc.", "inhibitory": "Inh."}
    terminal_endpoint_label = {"mt": "MT", "rrp": "RRP", "resource": "Res."}
    terminal_labels = [f"{terminal_pulse_label[p]}\n{terminal_endpoint_label[e]}" for p, e in terminal_columns]
    terminal_matrix = np.full((10, 9), np.nan)
    terminal_rows = []
    field_map = {
        "mt": "route_mt_support_delta",
        "rrp": "route_rrp_delta",
        "resource": "route_replenishment_resource_delta",
    }
    for i, route in enumerate(POP_ORDER):
        for j, (pulse_name, endpoint_short) in enumerate(terminal_columns):
            match = r4.loc[(r4["intended_route"] == route) & (r4["pulse"] == pulse_name)].iloc[0]
            prefix = field_map[endpoint_short]
            value = match[f"{prefix}_median"]
            terminal_matrix[i, j] = value
            terminal_rows.append(
                {
                    "intended_route": route,
                    "pulse": pulse_name,
                    "terminal_endpoint": endpoint_short,
                    "design_grid_median_delta": value,
                    "design_grid_range_min": match[f"{prefix}_range_min"],
                    "design_grid_range_max": match[f"{prefix}_range_max"],
                    "complete_cells": match[f"{prefix}_complete_cells"],
                }
            )
    save_table("Figure_7B_data.csv", pd.DataFrame(terminal_rows), table_registry, provenance)
    heatmap(axis_b, terminal_matrix, POP_ORDER, terminal_labels, "Route-local terminal engagement", diverging=True, colorbar_label="Median exact change")
    panel_label(axis_b, "B")
    matrices: list[np.ndarray] = []
    panel_records: list[tuple[str, str, str, np.ndarray, pd.DataFrame]] = []
    label_sequence = iter(["C", "D", "E", "F", "G", "H"])
    for row_index, axis_name in enumerate(["lr", "fe"]):
        for column_offset, pulse_name in enumerate(PULSE_ORDER, start=1):
            label = next(label_sequence)
            part = r5.loc[r5["pulse"] == pulse_name].copy()
            value_field = f"{axis_name}_nonadditivity_deg_median"
            matrix = matrix_from_long(part, "intended_class", "intended_route", value_field, POP_ORDER, POP_ORDER)
            matrices.append(matrix)
            panel = part[[
                "intended_class", "intended_route", "matrix_position", "pulse", "design_cell_count",
                "all_four_arms_rhythmic_cells", f"{axis_name}_nonadditivity_deg_complete_cells",
                value_field, f"{axis_name}_nonadditivity_deg_range_min", f"{axis_name}_nonadditivity_deg_range_max",
            ]].copy()
            panel_records.append((label, axis_name, pulse_name, matrix, panel))
    finite_values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices if np.isfinite(matrix).any()])
    shared_limit = max(float(np.max(np.abs(finite_values))), EPSILON)
    for index, (label, axis_name, pulse_name, matrix, panel) in enumerate(panel_records):
        row_index = 0 if axis_name == "lr" else 1
        column_index = PULSE_ORDER.index(pulse_name) + 1
        ax = fig.add_subplot(grid[row_index, column_index])
        save_table(f"Figure_7{label}_data.csv", panel, table_registry, provenance)
        heatmap(
            ax, matrix, POP_ORDER, POP_ORDER,
            f"{PULSE_LABEL[pulse_name]}: {axis_name.upper()} nonadditivity",
            diverging=True, vmin=-shared_limit, vmax=shared_limit,
            colorbar_label="Median exact interaction (deg)" if column_index == 3 else "",
            diagonal_outline=True,
        )
        ax.set_xlabel("Impaired route")
        if column_index == 1:
            ax.set_ylabel("Ablated class")
        else:
            ax.set_yticklabels([])
        panel_label(ax, label)
    fig.suptitle("Presynaptic route effects and class-by-route dependencies", fontsize=14, fontweight="bold")
    save_figure(fig, 7, figure_registry)

    # Figure 8: G epoch/window behavior and H timing/location/KCa controls.
    g_epoch = data["g_epoch"].copy()
    g_did = data["g_did"].copy()
    h_mt = data["h_mt"].copy()
    h_did = data["h_did"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=True)
    arm_order = ["no_challenge_intact", "no_challenge_impaired", "challenge_intact", "challenge_impaired"]
    arm_label = {
        "no_challenge_intact": "No challenge, intact",
        "no_challenge_impaired": "No challenge, impaired",
        "challenge_intact": "Challenge, intact",
        "challenge_impaired": "Challenge, impaired",
    }
    arm_color = {
        "no_challenge_intact": BLACK,
        "no_challenge_impaired": SKY,
        "challenge_intact": ORANGE,
        "challenge_impaired": VERMILION,
    }
    epoch_panels = [("A", "lr_phase_error_mean_abs_deg", "L-R phase error"), ("B", "fe_phase_error_mean_abs_deg", "F-E phase error")]
    for ax, (label, field, title) in zip(axes[0, :2], epoch_panels):
        records = []
        for arm in arm_order:
            for epoch in range(1, 25):
                subset = g_epoch.loc[(g_epoch["arm"] == arm) & (g_epoch["epoch"] == epoch), field]
                values = finite(subset)
                records.append(
                    {
                        "arm": arm,
                        "epoch": epoch,
                        "route_grid_cell_count": len(values),
                        "route_grid_median": np.median(values) if values else np.nan,
                        "route_grid_range_min": np.min(values) if values else np.nan,
                        "route_grid_range_max": np.max(values) if values else np.nan,
                    }
                )
        panel = pd.DataFrame(records)
        save_table(f"Figure_8{label}_data.csv", panel, table_registry, provenance)
        for start, end, color in [(1.5, 6.5, "#E8F1F8"), (6.5, 12.5, "#FFF3D6"), (12.5, 18.5, "#FBE5DF"), (19.5, 24.5, "#E4F3EC")]:
            ax.axvspan(start, end, color=color, zorder=0)
        for arm in arm_order:
            subset = panel.loc[panel["arm"] == arm]
            x = subset["epoch"].to_numpy(dtype=float)
            y = subset["route_grid_median"].to_numpy(dtype=float)
            lo = subset["route_grid_range_min"].to_numpy(dtype=float)
            hi = subset["route_grid_range_max"].to_numpy(dtype=float)
            ax.plot(x, y, color=arm_color[arm], linewidth=1.6, label=arm_label[arm])
            if int(subset["route_grid_cell_count"].max()) > 1:
                ax.fill_between(x, lo, hi, color=arm_color[arm], alpha=0.10, linewidth=0)
        ax.axvline(19, color=GRAY, linestyle=":", linewidth=0.9)
        ax.set_xlim(1, 24)
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Absolute phase error (deg)")
        ax.set_title(f"{title} across epochs")
        style_axes(ax)
        panel_label(ax, label)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    for axis_name, ax, label in [("lr", axes[0, 2], "C"), ("fe", axes[1, 0], "D")]:
        endpoint = f"{axis_name}_phase_error_mean_abs_deg"
        panel = g_did.loc[g_did["endpoint"] == endpoint, [
            "route", "window", "endpoint", "complete_pair", "all_arms_rhythmic", "contrast_value", "contrast_direction",
        ]].copy()
        save_table(f"Figure_8{label}_data.csv", panel, table_registry, provenance)
        matrix = matrix_from_long(panel, "route", "window", "contrast_value", POP_ORDER, WINDOW_ORDER)
        heatmap(
            ax, matrix, POP_ORDER, [WINDOW_LABEL[x] for x in WINDOW_ORDER],
            f"{axis_name.upper()} challenge-by-impairment effect", diverging=True,
            colorbar_label="Exact difference-in-differences (deg)",
        )
        panel_label(ax, label)
    mt_comparators = ["static_matched", "time_yoked", "spatial_shuffled"]
    mt_panel_rows = []
    row_keys = [(fast, comparator) for fast in FAST_ORDER for comparator in mt_comparators]
    column_keys = [(pulse_name, axis_name) for pulse_name in PULSE_ORDER for axis_name in ["lr", "fe"]]
    mt_matrix = np.full((len(row_keys), len(column_keys)), np.nan)
    for i, (fast, comparator) in enumerate(row_keys):
        for j, (pulse_name, axis_name) in enumerate(column_keys):
            endpoint = f"analysis_{axis_name}_phase_error_mean_abs_deg"
            part = h_mt.loc[
                (h_mt["fast_mode"] == fast)
                & (h_mt["mt_comparator"] == comparator)
                & (h_mt["pulse"] == pulse_name)
                & (h_mt["endpoint"] == endpoint)
            ]
            value = part.iloc[0]["contrast_value"]
            mt_matrix[i, j] = value
            mt_panel_rows.append(
                {
                    "fast_mode": fast,
                    "mt_comparator": comparator,
                    "pulse": pulse_name,
                    "phase_axis": axis_name.upper(),
                    "dynamic_minus_comparator_deg": value,
                    "complete_pair": part.iloc[0]["complete_pair"],
                    "failure_transition": part.iloc[0]["failure_transition_reference_to_target"],
                }
            )
    save_table("Figure_8E_data.csv", pd.DataFrame(mt_panel_rows), table_registry, provenance)
    heatmap(
        axes[1, 1], mt_matrix,
        [f"{FAST_LABEL[f].replace(' KCa', '')} | {MT_LABEL[m]}" for f, m in row_keys],
        [f"{PULSE_LABEL[p]}\n{a.upper()}" for p, a in column_keys],
        "Terminal-support timing and location controls", diverging=True,
        colorbar_label="Dynamic - comparator (deg)",
    )
    panel_label(axes[1, 1], "E")
    did_comparators = ["time_yoked", "spatial_shuffled"]
    did_rows = [(fast, comparator) for fast in ["static_mean", "yoked", "off"] for comparator in did_comparators]
    did_matrix = np.full((len(did_rows), len(column_keys)), np.nan)
    did_panel_rows = []
    for i, (fast_comparator, mt_comparator) in enumerate(did_rows):
        for j, (pulse_name, axis_name) in enumerate(column_keys):
            endpoint = f"analysis_{axis_name}_phase_error_mean_abs_deg"
            part = h_did.loc[
                (h_did["fast_comparator"] == fast_comparator)
                & (h_did["mt_comparator"] == mt_comparator)
                & (h_did["pulse"] == pulse_name)
                & (h_did["endpoint"] == endpoint)
            ]
            value = part.iloc[0]["exact_difference_in_differences"]
            did_matrix[i, j] = value
            did_panel_rows.append(
                {
                    "fast_comparator": fast_comparator,
                    "mt_comparator": mt_comparator,
                    "pulse": pulse_name,
                    "phase_axis": axis_name.upper(),
                    "exact_difference_in_differences_deg": value,
                    "complete_pair": part.iloc[0]["complete_pair"],
                }
            )
    save_table("Figure_8F_data.csv", pd.DataFrame(did_panel_rows), table_registry, provenance)
    heatmap(
        axes[1, 2], did_matrix,
        [f"{FAST_LABEL[f].replace(' KCa', '')} | {MT_LABEL[m]}" for f, m in did_rows],
        [f"{PULSE_LABEL[p]}\n{a.upper()}" for p, a in column_keys],
        "KCa modulation of timing/location contrasts", diverging=True,
        colorbar_label="Exact difference-in-differences (deg)",
    )
    panel_label(axes[1, 2], "F")
    fig.suptitle("Composite-stress dynamics and terminal-support specificity", fontsize=14, fontweight="bold")
    save_figure(fig, 8, figure_registry)


def captions() -> dict[str, dict[str, str]]:
    common = (
        "All values are conditional on one frozen stochastic network realization. "
        "Medians and ranges summarize the prespecified design grid and are not sampling-uncertainty estimates."
    )
    return {
        "Figure 4": {
            "title": "Intact-network phase stability across descending drive and mechanical load.",
            "caption": (
                "Heat maps show exact values in all nine no-pulse descending-drive-by-load cells: (A) L-R and (B) F-E absolute phase error, "
                "and (C) L-R and (D) F-E slip fractions with raw slip/cycle counts retained in the panel data. "
                "All nine conditions remained rhythmic. The 45-degree criterion is the frozen slip threshold. " + common
            ),
        },
        "Figure 5": {
            "title": "Pulse-evoked phase change and full-follow-up recovery in the intact network.",
            "caption": (
                "Exact post-pulse minus direction-matched sham differences are shown for (A) L-R and (B) F-E phase error across all descending-drive-by-load cells; "
                "horizontal bars are design-grid medians. (C) Recovery endpoint eligibility and event status are reported separately from phase effects. "
                "(D) Times are shown only for observed recovery events; full-follow-up censor fields remain in the panel data. Marker shape denotes the low, medium, or high descending-drive command and color denotes load. " + common
            ),
        },
        "Figure 6": {
            "title": "Circuit interventions and recruitment across descending-drive commands.",
            "caption": (
                "Design-grid medians of exact context-matched phase effects are shown for (A) ten single interventions and (B) six paired interventions. "
                "(C) Paired-intervention nonadditivity is pair - single 1 - single 2 + intact; positive values indicate a larger phase burden than the additive expectation. "
                "(D) Exact intact-network firing rates show recruitment across descending-drive and pulse contexts. The command level is an experimental input and is not treated as measured locomotor speed or assumed to yield monotonic network frequency. Failure and recovery counts are reported separately in Table R3. " + common
            ),
        },
        "Figure 7": {
            "title": "Presynaptic route effects and class-by-route dependencies.",
            "caption": (
                "(A) Route impairment minus exact matched intact phase effects and (B) route-local changes in phenomenological terminal support, readily releasable pool, and replenishment resource. "
                "(C-H) Full 10-by-10 class-by-route nonadditivity matrices for L-R and F-E phase error under no-pulse, excitatory, and inhibitory contexts. "
                "Outlined cells are diagonal class-route combinations; gray cells are undefined because all four rhythmic arms were not available. "
                "Terminal-state measures explain model mechanism but are not evidence for microtubule biology. " + common
            ),
        },
        "Figure 8": {
            "title": "Composite-stress dynamics and terminal-support specificity.",
            "caption": (
                "Epoch trajectories for (A) L-R and (B) F-E phase error show route-grid medians and full ranges; the shared no-challenge intact arm is a single trajectory. "
                "Shading marks the locked baseline, stress pre-challenge, stress-plus-challenge, and recovery windows; epochs 1 and 19 are display-only transitions. "
                "Exact route-level challenge-by-impairment effects are shown for (C) L-R and (D) F-E phase error. "
                "(E) Dynamic terminal support is compared with static-matched, time-yoked, and spatially shuffled controls within each KCa mode. "
                "(F) Exact difference-in-differences values show KCa modulation of time- and location-specific contrasts. " + common
            ),
        },
    }


def table_captions() -> dict[str, dict[str, str]]:
    common = (
        "Counts and summaries are conditional on one frozen stochastic realization. "
        "Design cells, tasks, routes, and reused simulation identifiers are not independent replicates."
    )
    return {
        "Table R1": {
            "title": "Frozen A-H experiment inventory and analysis rules.",
            "note": "Task and physical-simulation counts, validity, biological rhythmic failure, pulse delivery, and the locked comparison unit are reported separately for each stage. " + common,
        },
        "Table R2": {
            "title": "Intact-network phase stability and pulse-recovery summary.",
            "note": "The nine no-pulse descending-drive-by-load cells are exact values. The low/medium/high field is an input command, not measured locomotor speed. Active-pulse phase effects are post-pulse minus direction-matched sham; recovery eligibility, event status, and observed times are separate fields. " + common,
        },
        "Table R3": {
            "title": "Context-matched single and paired intervention effects.",
            "note": "No-pulse rows use whole-run phase endpoints and active-pulse rows use post-pulse endpoints. Pair nonadditivity is pair - single 1 - single 2 + intact. Medians and ranges summarize nine prespecified descending-drive-by-load cells. " + common,
        },
        "Table R4": {
            "title": "Effects of ten presynaptic route impairments.",
            "note": "Each route is compared with its exact intact descending-drive-by-load-by-pulse reference. Functional phase, propagation, biological failure/recovery, and terminal-state changes are kept distinct. " + common,
        },
        "Table R5": {
            "title": "Class-by-route marginal effects and exact nonadditivity.",
            "note": "All 10-by-10 class-route cells are reported for each pulse. Continuous summaries remain blank where the required rhythmic arms are unavailable; blank values are not recoded as neutral. " + common,
        },
        "Table R6": {
            "title": "Composite-stress and terminal-support/KCa mechanism controls.",
            "note": "Stage G uses exact challenge-by-impairment differences-in-differences by locked window. Stage H reports dynamic-minus-terminal-support controls, KCa-control-minus-dynamic-KCa contrasts, and their exact differences-in-differences. Recovery and rhythmic failure remain separate from continuous phase effects. " + common,
        },
    }


def write_caption_files(caption_map: dict[str, dict[str, str]], table_map: dict[str, dict[str, str]]) -> dict[str, Any]:
    json_path = FIGURE_DIR / "FIGURE_CAPTIONS.json"
    md_path = FIGURE_DIR / "FIGURE_CAPTIONS.md"
    json_path.write_text(json.dumps(caption_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Draft captions for Figures 4-8", ""]
    for figure, content in caption_map.items():
        lines.extend([f"## {figure}. {content['title']}", "", content["caption"], ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    table_json_path = TABLE_DIR / "TABLE_TITLES_AND_NOTES.json"
    table_md_path = TABLE_DIR / "TABLE_TITLES_AND_NOTES.md"
    table_json_path.write_text(json.dumps(table_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table_lines = ["# Draft titles and notes for Tables R1-R6", ""]
    for table, content in table_map.items():
        table_lines.extend([f"## {table}. {content['title']}", "", content["note"], ""])
    table_md_path.write_text("\n".join(table_lines), encoding="utf-8")
    return {
        json_path.name: {"bytes": json_path.stat().st_size, "sha256": sha256_file(json_path)},
        md_path.name: {"bytes": md_path.stat().st_size, "sha256": sha256_file(md_path)},
        table_json_path.name: {"bytes": table_json_path.stat().st_size, "sha256": sha256_file(table_json_path)},
        table_md_path.name: {"bytes": table_md_path.stat().st_size, "sha256": sha256_file(table_md_path)},
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    script_hash = sha256_file(SCRIPT_PATH)
    spec_hash = sha256_file(SPEC_PATH)
    protocol_hash = sha256_file(LOCKED_PROTOCOL_PATH)
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    master_qc = json.loads(MASTER_QC_PATH.read_text(encoding="utf-8"))
    if not master_qc.get("all_checks_pass"):
        raise RuntimeError("Master derived-output QC is not passing")
    if spec_hash != master_qc["protocol_spec_sha256"] or protocol_hash != master_qc["locked_protocol_sha256"]:
        raise RuntimeError("Locked protocol provenance does not match master QC")
    analysis_hashes = master_qc["analysis_script_sha256"]
    provenance = {
        "generated_at": generated_at,
        "script_hash": script_hash,
        "spec_hash": spec_hash,
        "protocol_hash": protocol_hash,
        "analysis_hashes": analysis_hashes,
    }
    sources = {
        "a_grid": read_csv("a_to_d_a_intact_no_pulse_grid.csv"),
        "a_pulse": read_csv("a_to_d_a_pulse_vs_matched_sham.csv"),
        "a_inventory": read_csv("a_to_d_stage_inventory.csv"),
        "b_cont": read_csv("a_to_d_b_single_continuous_effects.csv"),
        "b_states": read_csv("a_to_d_b_single_states.csv"),
        "c_cont": read_csv("a_to_d_c_pair_continuous_effects.csv"),
        "c_states": read_csv("a_to_d_c_pair_states.csv"),
        "recruitment": read_csv("a_to_d_d_recruitment_rates.csv"),
        "e_phase": read_csv("e_f_e_phase_recovery_atomic.csv"),
        "e_prop": read_csv("e_f_e_propagation_atomic.csv"),
        "e_terminal": read_csv("e_f_e_terminal_state_atomic.csv"),
        "f_phase": read_csv("e_f_f_phase_recovery_atomic.csv"),
        "g_epoch": read_csv("g_h_g_epoch_atomic.csv"),
        "g_did": read_csv("g_h_g_route_did.csv"),
        "h_conditions": read_csv("g_h_h_conditions.csv"),
        "h_mt": read_csv("g_h_h_mt_contrasts.csv"),
        "h_fast": read_csv("g_h_h_fast_contrasts.csv"),
        "h_did": read_csv("g_h_h_mt_by_fast_did.csv"),
    }
    # Ensure final derived sources have not changed since the master audit.
    source_checks: dict[str, bool] = {}
    for name, audit in master_qc["audited_csv"].items():
        path = DERIVED_DIR / name
        source_checks[f"derived_hash::{name}"] = path.exists() and sha256_file(path) == audit["sha256"]
    if not all(source_checks.values()):
        raise RuntimeError("One or more derived source hashes changed after master QC")

    table_registry: dict[str, dict[str, Any]] = {}
    figure_registry: dict[str, dict[str, Any]] = {}
    r1 = make_r1(spec, sources["a_inventory"], sources["e_phase"], sources["f_phase"], sources["g_epoch"], sources["h_conditions"])
    r2 = make_r2(sources["a_grid"], sources["a_pulse"])
    r3 = make_r3(sources["b_cont"], sources["b_states"], sources["c_cont"], sources["c_states"])
    r4 = make_r4(sources["e_phase"], sources["e_prop"], sources["e_terminal"])
    r5 = make_r5(sources["f_phase"])
    r6 = make_r6(sources["g_did"], sources["h_mt"], sources["h_fast"], sources["h_did"])
    save_table("Table_R1_experiment_inventory.csv", r1, table_registry, provenance)
    save_table("Table_R2_intact_network_and_pulse_summary.csv", r2, table_registry, provenance)
    save_table("Table_R3_single_and_paired_interventions.csv", r3, table_registry, provenance)
    save_table("Table_R4_presynaptic_route_effects.csv", r4, table_registry, provenance)
    save_table("Table_R5_class_by_route_dependencies.csv", r5, table_registry, provenance)
    save_table("Table_R6_composite_stress_and_terminal_support_controls.csv", r6, table_registry, provenance)
    sources.update({"r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5, "r6": r6})
    build_figures(sources, table_registry, figure_registry, provenance)
    caption_registry = write_caption_files(captions(), table_captions())

    forbidden = set(spec["forbidden_inference_fields"])
    forbidden_hits: dict[str, list[str]] = {}
    composite_hits: dict[str, list[str]] = {}
    for filename in sorted(table_registry):
        fields = list(pd.read_csv(TABLE_DIR / filename, nrows=0).columns)
        hits = sorted(set(fields) & forbidden)
        if hits:
            forbidden_hits[filename] = hits
        composite = [field for field in fields if "recovery_composite" in field.lower()]
        if composite:
            composite_hits[filename] = composite
    required_tables = [f"Table_R{i}_" for i in range(1, 7)]
    checks: dict[str, bool] = {
        **source_checks,
        "master_analysis_qc_pass": bool(master_qc["all_checks_pass"]),
        "six_main_tables_present": all(any(name.startswith(prefix) for name in table_registry) for prefix in required_tables),
        "all_panel_data_csv_present": all(
            any(name.startswith(f"Figure_{number}{panel}_data") for name in table_registry)
            for number, panels in {4: "ABCD", 5: "ABCD", 6: "ABCD", 7: "ABCDEFGH", 8: "ABCDEF"}.items()
            for panel in panels
        ),
        "no_forbidden_inference_fields": not forbidden_hits,
        "no_composite_recovery_fields": not composite_hits,
        "figure_4_to_8_png_and_pdf_present": all(f"Figure_{number}" in figure_registry for number in range(4, 9)),
        "png_300_dpi": all(
            299 <= info["png"]["dpi_x"] <= 301 and 299 <= info["png"]["dpi_y"] <= 301
            for info in figure_registry.values()
        ),
        "png_minimum_2400_px_width": all(info["png"]["width_px"] >= 2400 for info in figure_registry.values()),
        "pdf_single_page": all(info["pdf"]["pages"] == 1 for info in figure_registry.values()),
        "figure_and_table_caption_json_and_markdown_present": len(caption_registry) == 4,
        "R1_stage_counts_match_protocol": r1.set_index("stage")["task_count"].to_dict() == spec["expected_counts"]["stage_tasks"],
        "R1_F_biological_failure_counts_exact": int(r1.set_index("stage").loc["F", "rhythmic_failure_task_or_simulation_count"]) == 540
        and int(r1.set_index("stage").loc["F", "unique_rhythmic_failure_simulation_count"]) == 297,
        "R1_F_pulse_delivery_counts_exact": int(r1.set_index("stage").loc["F", "pulse_required_task_count"]) == 7200
        and int(r1.set_index("stage").loc["F", "pulse_delivered_task_count"]) == 6840,
        "R2_nine_intact_cells": int((r2["record_type"] == "intact_no_pulse_exact_cell").sum()) == 9,
        "R3_expected_summary_rows": len(r3) == 48,
        "R4_expected_route_pulse_rows": len(r4) == 30,
        "R5_full_10x10x3_rows": len(r5) == 300,
        "R6_expected_rows": len(r6) == 199,
        "failure_and_recovery_fields_separate": all(
            not ("failure" in field.lower() and "recovery" in field.lower())
            for frame in [r2, r3, r4, r5, r6]
            for field in frame.columns
        ),
    }
    all_pass = all(checks.values())
    qc = {
        "schema": "cpg-publication-output-qc-1.0",
        "generated_at_utc": generated_at,
        "script": str(SCRIPT_PATH.relative_to(ANALYSIS_DIR.parent)),
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_hash,
        "protocol_spec_sha256": spec_hash,
        "locked_protocol_sha256": protocol_hash,
        "analysis_script_sha256": analysis_hashes,
        "inference_scope": "conditional on one frozen realization; no sampling inference",
        "source_derived_files": {name: master_qc["audited_csv"][name] for name in master_qc["audited_csv"]},
        "publication_tables_and_panel_data": table_registry,
        "publication_figures": figure_registry,
        "caption_files": caption_registry,
        "forbidden_inference_field_hits": forbidden_hits,
        "composite_recovery_field_hits": composite_hits,
        "checks": checks,
        "all_checks_pass": all_pass,
    }
    qc_path = QA_DIR / "publication_outputs_qc.json"
    qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_lines = [
        f"generated_at_utc={generated_at}",
        f"script_sha256={script_hash}",
        f"tables_and_panel_csv={len(table_registry)}",
        f"figures={len(figure_registry)}",
        f"all_checks_pass={all_pass}",
    ]
    for name, info in figure_registry.items():
        log_lines.append(
            f"{name}: png={info['png']['width_px']}x{info['png']['height_px']}@{info['png']['dpi_x']:.3f}dpi "
            f"({info['png']['bytes']} bytes), pdf_pages={info['pdf']['pages']} ({info['pdf']['bytes']} bytes)"
        )
    (QA_DIR / "build_publication_outputs_run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    if not all_pass:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Publication-output QC failed: {failed}")
    print(json.dumps({"status": "PASS", "tables_and_panel_csv": len(table_registry), "figures": len(figure_registry), "qc": str(qc_path)}, indent=2))


if __name__ == "__main__":
    main()
