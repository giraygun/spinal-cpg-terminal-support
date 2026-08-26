#!/usr/bin/env python3
"""Causally separated, biologically constrained spinal CPG model, version 2.5.

Version 2.5 resolves RG, PF, MN, V0D, V0V, V2a, V3, V1Ia, V1Ren and V2b
spiking populations on both sides and for flexor/extensor phases. ``PF`` is
explicitly a functional premotor module rather than a genetic cell class.
For every named biological population, the full dynamical identity comprises
the membrane core, evidence-supported intrinsic terms (where they are known),
receptor/input dynamics, transmitter sign and projection equation. The code
does not invent class-specific ion channels for populations for which such
data do not exist. V2a and V3 heterogeneity and Renshaw-cell Ih/SK dynamics are
implemented rather than represented by labels. Every chemical projection is
a sparse neuron-to-neuron graph with an explicit delay. A minimal antagonistic
neuromuscular plant explicitly represents MN terminal vesicle availability,
ACh/nAChR transmission, end-plate potential, muscle Ca2+, cross-bridge
activation and force. It generates separate length/velocity (Ia-like) and
force (Ib-like) channels. Load changes mechanics, never afferent current directly.

No cell receives target frequency, desired phase, phase error, recovery error,
or a q-like control signal. Speed is a single descending command applied only
to the rhythmogenic module; there are no class-signed speed offsets.
Microtubule tracks are local to every central presynaptic terminal. Activity
can nucleate tracks and tracks can alter only the slow component of synaptic
vesicle replenishment. MT never senses phase, changes noise, rescales a fixed
weight, repairs damage or enters any primary endpoint. The long challenge
externally depletes identical RRP/slow-resource states; its primary endpoint is downstream
RG-to-MN transfer failure, never a vesicle or MT state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.special import expit


SIDES = ("L", "R")
PHASES = ("F", "E")
CLASSES = (
    "RG", "PF", "MN", "V0D", "V0V", "V2a", "V3", "V1Ia", "V1Ren", "V2b",
)
SIDE_PHASES = (("L", "F"), ("L", "E"), ("R", "F"), ("R", "E"))
SPEED_LEVELS = ("low", "medium", "high")
LOAD_CONTEXTS = ("normal", "unilateral", "bilateral_high")
PULSE_DIRECTIONS = ("none", "excitatory", "inhibitory")
MT_ROUTES = CLASSES
AFFERENT_ABLATIONS = ("Ia", "Ib", "groupI")
MT_MODES = (
    "dynamic", "static_matched", "time_yoked", "spatial_shuffled",
    "impaired", "off",
)
MODEL_VERSION = "distributed-local-terminal-mt-cpg-2.6.1-candidate"


def population_name(cell_class: str, side: str, phase: str) -> str:
    return f"{cell_class}_{side}_{phase}"


POPULATIONS = tuple(
    population_name(cell_class, side, phase)
    for cell_class in CLASSES
    for side, phase in SIDE_PHASES
)
POP_INDEX = {name: index for index, name in enumerate(POPULATIONS)}


def pop(cell_class: str, side: str, phase: str) -> int:
    return POP_INDEX[population_name(cell_class, side, phase)]


def other_side(side: str) -> str:
    return "R" if side == "L" else "L"


def other_phase(phase: str) -> str:
    return "E" if phase == "F" else "F"


@dataclass(frozen=True)
class Config:
    dt_ms: float = 0.025
    duration_s: float = 10.0
    burn_in_s: float = 2.0
    rg_neurons: int = 16
    relay_neurons: int = 8
    pf_neurons: int = 12
    mn_neurons: int = 12

    # A single experimentally interpretable descending command changes RG
    # excitation. It contains no target frequency and no cell-class-specific
    # sign. All interneuron recruitment must arise downstream from the network.
    descending_rg_drive_offsets_pa: Tuple[float, ...] = (-18.0, 0.0, 22.0)

    # Shared AdEx voltage landmarks. Capacitance, leak, adaptation and tonic
    # drive are deliberately *not* shared: they live in CELL_CLASS_EQUATIONS.
    leak_reversal_mv: float = -65.0
    threshold_mv: float = -50.0
    slope_factor_mv: float = 2.0
    reset_mv: float = -58.0
    spike_peak_mv: float = 20.0
    refractory_ms: float = 2.0
    # Neonatal P2--P5 Shox2-current prevalences are rounded once across the
    # full modeled RG class, then allocated across L/R x F/E contexts without
    # reading outcomes. They are sample-derived structural priors, not claims
    # that all currents co-occur in one biological subtype.
    # The adult pharmacology establishing NaP and L-type-Ca PIC components is
    # transferred only as a qualitative component identity; their shared mask,
    # exact gates and gains remain explicit H-level reductions.
    rg_pic_positive_fraction: float = 27.0 / 35.0
    rg_m_positive_fraction: float = 31.0 / 35.0
    rg_kca_positive_fraction: float = 1.0 / 35.0
    rg_h_positive_fraction: float = 20.0 / 35.0
    rg_t_positive_fraction: float = 6.0 / 35.0
    rg_a_positive_fraction: float = 5.0 / 35.0
    # Paired recordings found electrical coupling in 6/18 nearby neonatal
    # Shox2+ non-V2a pairs. This is a pair probability—not a cell prevalence.
    # A symmetric sparse within-population graph realizes that local-pair prior.
    rg_gap_pair_probability: float = 6.0 / 18.0
    rg_nap_conductance_ns: float = 4.7
    sodium_reversal_mv: float = 55.0
    nap_activation_half_mv: float = -47.0
    nap_activation_slope_mv: float = 4.5
    nap_inactivation_half_mv: float = -48.0
    nap_inactivation_slope_mv: float = 5.0
    nap_inactivation_tau_ms: float = 560.0
    rg_ltype_ca_pic_conductance_ns: float = 0.20
    rg_ltype_ca_pic_activation_half_mv: float = -45.0
    rg_ltype_ca_pic_activation_slope_mv: float = 5.5
    rg_ltype_ca_pic_activation_tau_ms: float = 80.0
    rg_ltype_ca_reversal_mv: float = 120.0

    # Additional neonatal Shox2 subset currents measured in the same primary
    # study. Molecular subtype, exact gates and conductances are not inferred.
    rg_h_conductance_ns: float = 0.20
    rg_h_reversal_mv: float = -35.0
    rg_h_half_mv: float = -72.0
    rg_h_slope_mv: float = 6.0
    rg_h_tau_ms: float = 180.0
    rg_t_conductance_ns: float = 0.15
    rg_t_reversal_mv: float = 120.0
    rg_t_activation_half_mv: float = -58.0
    rg_t_activation_slope_mv: float = 6.0
    rg_t_inactivation_half_mv: float = -78.0
    rg_t_inactivation_slope_mv: float = 5.0
    rg_t_inactivation_tau_ms: float = 45.0
    rg_a_conductance_ns: float = 0.20
    rg_a_activation_half_mv: float = -30.0
    rg_a_activation_slope_mv: float = 8.0
    rg_a_inactivation_half_mv: float = -60.0
    rg_a_inactivation_slope_mv: float = 6.0
    rg_a_inactivation_tau_ms: float = 20.0
    rg_gap_conductance_ns: float = 0.08

    # RG-restricted effective M-current. Its conductance is invariant across
    # speed contexts; speed is not written through a second hidden control.
    rg_m_conductance_ns: float = 25.0
    m_activation_half_mv: float = -42.0
    m_activation_slope_mv: float = 7.0
    m_activation_tau_ms: float = 90.0
    rg_m_conductance_scale: float = 0.65

    # Activity-dependent RG effective KCa/slow-afterhyperpolarizing current. "Phase
    # corrector" is intentionally not used: phase stabilization is an outcome.
    kca_conductance_ns: float = 4.0
    potassium_reversal_mv: float = -85.0
    calcium_decay_ms: float = 150.0
    calcium_spike_increment: float = 0.18
    calcium_half_activation: float = 0.32
    calcium_hill_coefficient: float = 4.0
    static_kca_activation_reference: float = 0.0003223102848596752

    # PF is a functional premotor module. This slow integration state is an
    # explicit coarse-graining assumption, not a claim of PF-specific NMDA
    # receptor expression or of a molecular PF cell identity.
    pf_slow_integration_conductance_ns: float = 0.10
    pf_slow_integration_half_pa: float = 45.0
    pf_slow_integration_rise_ms: float = 18.0
    pf_slow_integration_decay_ms: float = 90.0
    excitatory_reversal_mv: float = 0.0

    # V2a population heterogeneity measured in neonatal mouse: tonic, phasic
    # and delayed-onset firing phenotypes. The fractions are phenotype priors,
    # not claims of immutable adult proportions.
    v2a_variant_fractions: Tuple[float, ...] = (
        82.0 / 121.0, 29.0 / 121.0, 10.0 / 121.0,
    )
    # The delayed-onset phenotype is measured, but its unique channel identity
    # is not established. This effective transient outward state therefore
    # carries no molecular IA claim.
    v2a_delayed_onset_conductance_ns: float = 2.0
    v2a_delayed_activation_half_mv: float = -52.0
    v2a_delayed_activation_slope_mv: float = 6.0
    v2a_delayed_relief_tau_ms: float = 55.0
    v2a_phasic_adaptation_multiplier: float = 2.4
    v2a_gap_conductance_ns: float = 0.08
    # Pair-level electrical coupling incidence: tonic 13/47, phasic 3/7;
    # mixed tonic-phasic 0/7. Delayed pairs were not sampled and are omitted.
    v2a_tonic_gap_pair_probability: float = 13.0 / 47.0
    v2a_phasic_gap_pair_probability: float = 3.0 / 7.0
    # Ih-consistent sag prevalence measured within tonic/phasic/delayed V2a
    # phenotypes (51/82, 26/29 and 10/10). Applying those sample prevalences as
    # deterministic population priors and the exact gate kinetics remain H-level.
    v2a_h_positive_fractions: Tuple[float, ...] = (
        51.0 / 82.0, 26.0 / 29.0, 1.0,
    )
    v2a_h_conductance_ns: float = 0.20
    v2a_h_reversal_mv: float = -35.0
    v2a_h_half_mv: float = -72.0
    v2a_h_slope_mv: float = 6.0
    v2a_h_tau_ms: float = 180.0

    # V3 contains ventral fast-tonic and dorsal adapting/rebound phenotypes.
    # Ih is present in both measured phenotypes and is stronger dorsally;
    # low-threshold T-type Ca is restricted to the dorsal-like fraction.
    v3_dorsal_fraction: float = 0.50
    v3_ventral_h_conductance_ns: float = 0.175
    v3_dorsal_h_conductance_multiplier: float = 2.0
    v3_h_reversal_mv: float = -35.0
    v3_h_half_mv: float = -72.0
    v3_h_slope_mv: float = 6.0
    v3_h_tau_ms: float = 180.0
    v3_t_conductance_ns: float = 0.20
    v3_t_reversal_mv: float = 120.0
    v3_t_activation_half_mv: float = -58.0
    v3_t_activation_slope_mv: float = 6.0
    v3_t_inactivation_half_mv: float = -78.0
    v3_t_inactivation_slope_mv: float = 5.0
    v3_t_inactivation_tau_ms: float = 45.0
    v3_dorsal_adaptation_multiplier: float = 2.0

    # Chrna2-positive Renshaw-cell mechanisms: separate effective cholinergic
    # and glutamatergic MN-collateral conductances, hyperpolarization-activated
    # Ih and calcium-dependent SK/AHP current. The two effective receptor
    # mixtures are not claims about individual receptor-subunit kinetics.
    renshaw_nachr_decay_ms: float = 12.0
    renshaw_glutamate_decay_ms: float = 8.0
    renshaw_h_conductance_ns: float = 0.45
    renshaw_h_reversal_mv: float = -35.0
    renshaw_h_half_mv: float = -72.0
    renshaw_h_slope_mv: float = 6.0
    renshaw_h_tau_ms: float = 140.0
    renshaw_sk_conductance_ns: float = 0.55
    renshaw_calcium_decay_ms: float = 120.0
    renshaw_calcium_spike_increment: float = 0.18
    renshaw_calcium_half_activation: float = 0.28

    # Alpha-motoneuron soma/dendrite reduction: distinct dendritic persistent
    # Na-like and L-type Ca-like inward currents, reciprocal axial coupling and
    # somatic spike-triggered AHP. Exact gates/gains are reduced-model priors;
    # the experimentally distinct PIC families are not collapsed into one
    # sodium-reversal current.
    mn_dendrite_capacitance_pf: float = 120.0
    mn_dendrite_leak_ns: float = 5.0
    mn_soma_dendrite_coupling_ns: float = 4.0
    mn_dendritic_synaptic_fraction: float = 0.50
    mn_nap_pic_conductance_ns: float = 0.07
    mn_nap_pic_activation_half_mv: float = -48.0
    mn_nap_pic_activation_slope_mv: float = 5.0
    mn_nap_pic_inactivation_half_mv: float = -30.0
    mn_nap_pic_inactivation_slope_mv: float = 7.0
    mn_nap_pic_inactivation_tau_ms: float = 250.0
    mn_ltype_ca_pic_conductance_ns: float = 0.06
    mn_ltype_ca_pic_activation_half_mv: float = -45.0
    mn_ltype_ca_pic_activation_slope_mv: float = 5.5
    mn_ltype_ca_pic_activation_tau_ms: float = 80.0
    mn_ltype_ca_reversal_mv: float = 120.0
    mn_ahp_conductance_ns: float = 0.45
    mn_calcium_decay_ms: float = 190.0
    mn_calcium_spike_increment: float = 0.14
    mn_calcium_half_activation: float = 0.30

    # KCa is an optional cellular negative-feedback control. It is deliberately
    # independent of every MT state in version 2.5.

    # Tonic drive is class-specific in CELL_CLASS_EQUATIONS.
    drive_heterogeneity_fraction: float = 0.06

    # Noise
    noise_tau_ms: float = 5.0
    independent_noise_sigma_pa: float = 6.0
    population_common_noise_sigma_pa: float = 10.0
    noise_burst_multiplier: float = 2.0
    perturbation_start_s: float = 5.0
    perturbation_end_s: float = 5.75
    phase_kick_current_pa: float = 260.0
    phase_kick_duration_ms: float = 80.0
    excitatory_pulse_current_pa: float = 260.0
    inhibitory_pulse_current_pa: float = -320.0
    pulse_duration_ms: float = 80.0
    pulse_arm_after_s: float = 4.0
    excitatory_pulse_cycle_fraction: float = 0.20
    inhibitory_pulse_cycle_fraction: float = 0.20
    pf_deletion_current_pa: float = -900.0

    # Conductance-based central synapses. The pathway parameters below retain
    # pA units as peak-current-at-reference-voltage calibration quantities and
    # are converted once to positive receptor conductances at release.
    excitatory_synapse_decay_ms: float = 18.0
    inhibitory_synapse_decay_ms: float = 24.0
    inhibitory_reversal_mv: float = -75.0
    synaptic_reference_voltage_mv: float = -60.0
    rg_recurrent_excitation_pa: float = 210.0
    rg_to_v1_pa: float = 600.0
    v1_to_antagonist_rg_pa: float = 1500.0
    rg_to_v2a_pa: float = 300.0
    rg_to_v0d_pa: float = 500.0
    v2a_to_v0v_pa: float = 300.0
    rg_to_v3_pa: float = 250.0
    v0d_cross_inhibition_pa: float = 900.0
    v0v_to_cross_v1_pa: float = 420.0
    # Zhang 2008 supports predominantly contralateral V3 input to both
    # flexor and extensor motor pools. The exact current, probability, delay
    # and online phase assignment remain H-level reduced-model priors.
    v3_to_contralateral_mn_pa: float = 5.0
    # Chopek 2018 directly supports an ipsilateral ventrolateral-V3-to-MN
    # pathway and glutamatergic MN recurrence onto ventral V3. The fraction
    # and both peak-current-at-reference-voltage values below are explicit
    # H-level structural/strength priors, fixed before any model result.
    v3_vlat_fraction_of_ventral: float = 0.50
    v3_vlat_to_ipsilateral_mn_pa: float = 5.0
    mn_to_v3_vlat_glutamate_pa: float = 5.0
    rg_to_pf_pa: float = 450.0
    pf_recurrent_excitation_pa: float = 35.0
    v1_to_antagonist_pf_pa: float = 690.0
    rg_to_v2b_pa: float = 600.0
    v2b_to_antagonist_rg_pa: float = 1500.0
    v2b_to_antagonist_pf_pa: float = 690.0
    pf_to_mn_pa: float = 1200.0
    v1_to_antagonist_mn_pa: float = 780.0
    v2b_to_antagonist_mn_pa: float = 780.0
    # Britz et al. (2015) supports a graded V2b bias toward extensor motor
    # targets. This reduced-model ratio encodes only that direction; 0.5 is not
    # presented as a measured anatomical point estimate.
    v2b_flexor_target_relative_gain: float = 0.50
    # Equal splitting preserves the former total reference current but is an
    # H-level prior, not an estimate of the cholinergic/glutamatergic ratio.
    mn_to_v1ren_nachr_pa: float = 300.0
    mn_to_v1ren_glutamate_pa: float = 300.0
    v1ren_to_mn_pa: float = 300.0
    v1ren_to_v1ia_pa: float = 90.0
    v2a_to_pf_pa: float = 85.0
    v2a_to_mn_pa: float = 95.0
    synaptic_heterogeneity_fraction: float = 0.05

    # These are modeling priors, not claimed anatomical point estimates.
    recurrent_connection_probability: float = 0.50
    local_connection_probability: float = 0.50
    commissural_connection_probability: float = 0.50
    recurrent_delay_bins_ms: Tuple[float, ...] = (0.5, 1.0, 1.5)
    local_delay_bins_ms: Tuple[float, ...] = (1.0, 2.0, 3.0)
    commissural_delay_bins_ms: Tuple[float, ...] = (2.0, 4.0, 6.0)

    # Every central chemical terminal has a local activity, dynamic-MT,
    # readily releasable pool (RRP) and a normalized slow-replenishment
    # resource. This resource is not an anatomical vesicle reserve pool.
    # values are readout aggregates only; they never feed back as global state.
    mt_activity_tau_ms: float = 140.0
    mt_activity_spike_increment: float = 0.18
    mt_track_decay_ms: float = 2200.0
    mt_nucleation_per_ms: float = 0.00055
    mt_track_max: float = 1.0
    mt_initial_track_fraction: float = 0.08
    mt_static_route_supports: Tuple[float, ...] = (
        0.06711466909562264, 0.09832882287533674,
        0.10390085493050612, 0.04564195094109941,
        0.0475336012687683, 0.054125466877117,
        0.031234564085884854, 0.09431691160189962,
        0.12469218153508083, 0.06494781345848805,
    )
    mt_impaired_nucleation_scale: float = 0.12
    mt_impaired_lifetime_scale: float = 0.33
    vesicle_fast_recovery_ms: float = 95.0
    vesicle_slow_recovery_ms: float = 720.0
    slow_replenishment_resource_recovery_ms: float = 1800.0
    mt_slow_replenishment_gain: float = 6.0
    vesicle_depletion_fraction: float = 0.055
    long_rrp_challenge_floor: float = 0.20
    long_replenishment_resource_challenge_floor: float = 0.55
    challenge_route_fraction: float = 1.0

    # Explicit rate-coded neuromuscular junction (four L/R x F/E units).
    # The equations are mechanistic; gains are reduced-model priors rather than
    # direct point estimates transferred across species/developmental stages.
    # The abstract MN output event is already at the motor terminal. The fixed
    # 0.50-ms delay represents reduced synchronous release latency, not a
    # soma-to-muscle conduction-time point estimate.
    nmj_release_delay_ms: float = 0.50
    nmj_release_probability: float = 0.55
    nmj_vesicle_recovery_ms: float = 420.0
    nmj_ach_decay_ms: float = 2.2
    nmj_ach_release_gain: float = 3.0
    muscle_fiber_rest_mv: float = -85.0
    nmj_endplate_gain_mv: float = 78.0
    nmj_endplate_tau_ms: float = 3.0
    nmj_nachr_half_ach: float = 0.06
    nmj_nachr_hill: float = 1.5
    muscle_fiber_threshold_mv: float = -72.0
    muscle_fiber_slope_mv: float = 2.5

    # Excitation-contraction coupling and antagonistic mechanics.
    muscle_calcium_tau_ms: float = 32.0
    muscle_effective_sr_release_per_ms: float = 0.80
    muscle_calcium_half: float = 0.20
    # Konishi--Watanabe report intact-fibre Hill slopes 3.2--3.9; 3.5 is a
    # preregistered midpoint prior, not a fitted outcome-dependent value.
    muscle_calcium_hill: float = 3.5
    # Contextual load is an external viscous resistance, not a multiplier on
    # active muscle force. This keeps actuator production and environmental
    # resistance on opposite sides of the joint balance law.
    load_unilateral_resistance_multiplier: float = 1.35
    load_bilateral_high_resistance_multiplier: float = 1.45
    extensor_force_scale_prior: float = 1.20
    muscle_activation_tau_ms: float = 45.0
    joint_inertia: float = 1.0
    joint_damping: float = 3.2
    joint_stiffness: float = 5.5
    muscle_torque_gain: float = 4.0
    muscle_length_scale: float = 0.18
    ia_tonic: float = 0.20
    ia_length_gain: float = 2.4
    ia_velocity_gain: float = 0.18
    ib_tonic: float = 0.12
    ib_force_gain: float = 0.75
    ia_to_pf_pa: float = 18.0
    ia_to_mn_pa: float = 42.0
    ia_to_v1ia_pa: float = 46.0
    # The GTO force proxy and its central action are deliberately separate.
    # These positive gains implement one effective locomotor operating regime;
    # they are not a claim of monosynaptic Ib excitation or a fixed reflex sign.
    ib_effective_spinal_to_pf_pa: float = 24.0
    ib_effective_spinal_to_mn_pa: float = 34.0
    ib_effective_extensor_context_gain: float = 1.25
    sensory_resource_recovery_ms: float = 300.0
    sensory_resource_depletion_per_s: float = 0.035

    # Long demand -> exogenous vesicle-depletion challenge -> recovery. The
    # challenge is identical across MT conditions and contains no repair law.
    long_epoch_duration_s: float = 2.0
    long_n_epochs: int = 24
    long_baseline_end_epoch: int = 6
    long_demand_start_epoch: int = 7
    long_challenge_epoch: int = 13
    long_demand_end_epoch: int = 18

    # Readout and detection
    rate_tau_ms: float = 20.0
    burst_on_threshold_hz: float = 16.0
    burst_off_threshold_hz: float = 7.0
    pf_burst_on_threshold_hz: float = 14.0
    pf_burst_off_threshold_hz: float = 6.0
    mn_burst_on_threshold_hz: float = 10.0
    mn_burst_off_threshold_hz: float = 4.0
    minimum_interburst_s: float = 0.22
    rg_pf_match_window_s: float = 0.25
    # A motor event before its RG anchor cannot be counted as RG->MN transfer.
    rg_mn_match_pre_window_s: float = 0.0
    rg_mn_match_post_window_s: float = 0.25
    phase_slip_threshold_deg: float = 45.0
    recovery_consecutive_cycles: int = 3
    recovery_frequency_tolerance_fraction: float = 0.25


@dataclass(frozen=True)
class CellClassEquation:
    """Auditable full dynamical identity actually instantiated in the model.

    ``intrinsic_terms`` may be shared when primary data do not support a
    class-specific channel. In that case, receptor/input and output equations
    must provide the experimentally supported distinction. This prevents the
    registry from manufacturing biology merely to obtain a unique identifier.
    """

    cell_class: str
    biological_counterpart: str
    representation_kind: str
    equation_id: str
    equation_family: str
    transmitter: str
    capacitance_pf: float
    leak_ns: float
    adaptation_a_ns: float
    adaptation_tau_ms: float
    adaptation_b_pa: float
    tonic_drive_pa: float
    intrinsic_terms: Tuple[str, ...]
    receptor_or_input_terms: Tuple[str, ...]
    input_terms: Tuple[str, ...]
    output_terms: Tuple[str, ...]
    differentiating_domain: str
    execution_contract_id: str
    evidence_level: str
    literature_urls: Tuple[str, ...]
    parameter_status: str = "reduced_model_prior_not_patch_clamp_point_estimate"
    target_specific_transmitters: Tuple[Tuple[str, str], ...] = ()

    def mechanistic_signature(self) -> Tuple[object, ...]:
        """Return the qualitative, executable identity—not fitted numbers."""
        return (
            self.equation_family,
            self.intrinsic_terms,
            self.receptor_or_input_terms,
            self.input_terms,
            self.output_terms,
            self.differentiating_domain,
            self.target_specific_transmitters,
        )


# Important: RG and PF are declared as functional ensembles, not genetic cell
# subclasses. All other entries map to a genetic or physiological population.
# The parameters preserve literature-constrained qualitative phenotypes; no
# table entry is claimed to be a direct cross-study patch-clamp estimate.
CELL_CLASS_EQUATIONS: Dict[str, CellClassEquation] = {
    "RG": CellClassEquation(
        cell_class="RG",
        biological_counterpart="Shox2-enriched rhythmogenic interneuron ensemble",
        representation_kind="molecularly_anchored_functional_ensemble",
        equation_id="E_RG_NEONATAL_SUBSET_CURRENTS_AND_GAP",
        equation_family=(
            "AdEx + prevalence-masked NaP/L-type-Ca PIC, M, KCa/sAHP, "
            "Ih, T-type-Ca and A-type-K currents + within-RG gap current"
        ),
        transmitter="glutamate",
        capacitance_pf=200.0, leak_ns=10.0, adaptation_a_ns=1.50,
        adaptation_tau_ms=320.0, adaptation_b_pa=80.0, tonic_drive_pa=150.0,
        intrinsic_terms=(
            "I_LEAK", "I_ADEX_EXP", "I_ADAPT_W",
            "I_NAP_RG", "I_LTYPE_CA_PIC_RG", "I_M_RG", "I_KCA_RG",
            "I_H_RG", "I_T_RG", "I_A_RG", "I_RG_GAP",
        ),
        receptor_or_input_terms=("single_descending_command", "recurrent_glutamatergic_input"),
        input_terms=("descending_drive", "recurrent_excitation", "V1Ia/V2b inhibition", "commissural_input"),
        output_terms=("RG_recurrent", "RG_to_PF", "RG_to_interneurons"),
        differentiating_domain="intrinsic_and_network",
        execution_contract_id="XC_RG",
        evidence_level="direct_qualitative_mechanism_plus_reduced_model_gain",
        literature_urls=(
            "https://doi.org/10.1113/JP287752",
            "https://doi.org/10.7554/eLife.42519",
            "https://doi.org/10.1016/j.neuron.2013.08.015",
        ),
    ),
    "PF": CellClassEquation(
        cell_class="PF",
        biological_counterpart="latent excitatory premotor pattern-formation module; not a genetic subtype",
        representation_kind="functional_module",
        equation_id="E_PF_SLOW_PREMOTOR_INTEGRATION",
        equation_family="non-rhythmogenic AdEx + effective slow premotor integration",
        transmitter="glutamate",
        capacitance_pf=154.0, leak_ns=9.1, adaptation_a_ns=0.85,
        adaptation_tau_ms=185.0, adaptation_b_pa=17.0, tonic_drive_pa=60.0,
        intrinsic_terms=("I_LEAK", "I_ADEX_EXP", "I_ADAPT_W", "I_PF_SLOW"),
        receptor_or_input_terms=("s_slow_PF_effective", "I_slow_PF_effective"),
        input_terms=("RG_drive", "V2a_drive", "Ia_afferent/effective_Ib_spinal_input", "V1Ia/V2b inhibition"),
        output_terms=("PF_recurrent", "PF_to_MN"),
        differentiating_domain="functional_synaptic_integration",
        execution_contract_id="XC_PF",
        evidence_level="functional_architecture_not_genetic_cell_identity",
        literature_urls=(
            "https://doi.org/10.1152/jn.00216.2005",
            "https://doi.org/10.1113/jphysiol.2012.240895",
        ),
    ),
    "MN": CellClassEquation(
        cell_class="MN", biological_counterpart="alpha motoneuron pool",
        representation_kind="physiological_population",
        equation_id="E_MN_2COMP_NAP_LTYPE_CA_PIC_SK_AHP",
        equation_family=(
            "two-compartment soma AdEx/SK-AHP + separate dendritic NaP and "
            "L-type Ca persistent inward currents"
        ),
        transmitter="acetylcholine",
        capacitance_pf=180.0, leak_ns=10.0, adaptation_a_ns=1.20,
        adaptation_tau_ms=240.0, adaptation_b_pa=22.0, tonic_drive_pa=80.0,
        intrinsic_terms=(
            "I_LEAK", "I_ADEX_EXP", "I_ADAPT_W", "I_MN_DEND_LEAK",
            "I_MN_NAP_PIC", "I_MN_LTYPE_CA_PIC",
            "I_MN_COUPLING_SOMA", "I_MN_AHP",
        ),
        receptor_or_input_terms=(
            "premotor_excitation",
            "group_I_afferents",
            "V3_motor_excitation",
        ),
        input_terms=(
            "PF/V2a/V3 excitation",
            "Ia_afferent/effective_Ib_spinal_input",
            "V1Ia/V1Ren/V2b inhibition",
        ),
        output_terms=(
            "NMJ_ACh_release",
            "MN_to_V1Ren_paired_effective_ACh_glutamate_components",
            "MN_to_V3_VLat_glutamatergic_recurrent_excitation",
        ),
        differentiating_domain="intrinsic_and_target_specific_motor_outputs",
        execution_contract_id="XC_MN",
        evidence_level="direct_experimental_mechanisms_reduced_two_compartment",
        literature_urls=(
            "https://doi.org/10.1038/s41598-017-04266-8",
            "https://doi.org/10.1016/j.celrep.2018.08.095",
            "https://doi.org/10.1046/j.1460-9568.2000.00055.x",
            "https://doi.org/10.1152/jn.00236.2003",
            "https://doi.org/10.1152/jn.1980.43.6.1700",
            "https://doi.org/10.1152/jn.01068.2006",
        ),
        target_specific_transmitters=((
            "MN_to_V1Ren_nAChR", "acetylcholine",
        ), (
            "MN_to_V1Ren_GluR", "glutamate",
        ), (
            "MN_to_V3_VLat_GluR", "glutamate",
        )),
    ),
    "V0D": CellClassEquation(
        cell_class="V0D", biological_counterpart="Dbx1-derived inhibitory commissural V0D interneurons",
        representation_kind="genetic_class", equation_id="E_V0D_INHIB_COMMISSURAL",
        equation_family="shared inhibitory AdEx core + RG-gated glycine/GABA commissural equation",
        transmitter="glycine_GABA",
        capacitance_pf=148.0, leak_ns=8.8, adaptation_a_ns=0.70,
        adaptation_tau_ms=170.0, adaptation_b_pa=14.0, tonic_drive_pa=20.0,
        intrinsic_terms=("I_LEAK", "I_ADEX_EXP", "I_ADAPT_W"),
        receptor_or_input_terms=("RG_to_V0D_glutamatergic_drive",),
        input_terms=("RG_drive_only_no_speed_offset",),
        output_terms=("V0D_contralateral_glycine_GABA_inhibition",),
        differentiating_domain="transmitter_projection_and_target",
        execution_contract_id="XC_V0D",
        evidence_level="direct_identity_transmitter_projection_no_unique_channel_claim",
        literature_urls=("https://doi.org/10.1016/S0896-6273(04)00249-1", "https://doi.org/10.1038/nature12286"),
    ),
    "V0V": CellClassEquation(
        cell_class="V0V", biological_counterpart="Evx1-positive excitatory commissural V0V interneurons",
        representation_kind="genetic_class", equation_id="E_V0V_GLUT_DISYNAPTIC",
        equation_family="shared excitatory AdEx core + V2a-gated crossed-disinhibitory equation",
        transmitter="glutamate",
        capacitance_pf=152.0, leak_ns=9.0, adaptation_a_ns=0.75,
        adaptation_tau_ms=176.0, adaptation_b_pa=15.0, tonic_drive_pa=85.0,
        intrinsic_terms=("I_LEAK", "I_ADEX_EXP", "I_ADAPT_W"),
        receptor_or_input_terms=("V2a_to_V0V_glutamatergic_drive",),
        input_terms=("V2a_drive_only_no_speed_offset",),
        output_terms=("V0V_crossed_excitation_of_inhibitory_interneuron",),
        differentiating_domain="disynaptic_commissural_topology",
        execution_contract_id="XC_V0V",
        evidence_level="direct_identity_transmitter_function_no_unique_channel_claim",
        literature_urls=(
            "https://doi.org/10.1016/S0896-6273(01)00213-6",
            "https://doi.org/10.1016/j.neuron.2008.08.009",
            "https://doi.org/10.1038/nature12286",
        ),
    ),
    "V2a": CellClassEquation(
        cell_class="V2a", biological_counterpart="Chx10-positive ipsilateral excitatory V2a interneurons",
        representation_kind="genetic_class_with_measured_firing_subphenotypes",
        equation_id="E_V2A_TONIC_PHASIC_DELAYED_IH",
        equation_family=(
            "heterogeneous AdEx: tonic/phasic/delayed + phenotype-specific "
            "Ih-positive masks + effective delayed-onset state + supported "
            "within-phenotype gap current"
        ),
        transmitter="glutamate",
        capacitance_pf=156.0, leak_ns=9.2, adaptation_a_ns=0.82,
        adaptation_tau_ms=190.0, adaptation_b_pa=17.0, tonic_drive_pa=85.0,
        intrinsic_terms=(
            "I_LEAK", "I_ADEX_EXP", "I_ADAPT_W",
            "I_V2A_H_TONIC", "I_V2A_H_PHASIC", "I_V2A_H_DELAYED",
            "I_V2A_DELAY", "I_V2A_GAP_TONIC", "I_V2A_GAP_PHASIC",
        ),
        receptor_or_input_terms=("RG_to_V2a_glutamatergic_drive",),
        input_terms=("RG_drive_only_no_speed_offset",),
        output_terms=("V2a_to_V0V", "V2a_to_PF", "V2a_to_MN"),
        differentiating_domain="measured_intrinsic_heterogeneity_and_ipsilateral_projection",
        execution_contract_id="XC_V2A",
        evidence_level=(
            "direct_firing_phenotypes_V0_and_population_level_MN_paths_supported_"
            "PF_target_exact_phenotype_contribution_and_effective_delay_are_H_level"
        ),
        literature_urls=(
            "https://doi.org/10.1523/JNEUROSCI.4849-09.2010",
            "https://doi.org/10.1016/j.neuron.2008.08.009",
            "https://doi.org/10.1016/j.neuron.2018.01.023",
        ),
    ),
    "V3": CellClassEquation(
        cell_class="V3", biological_counterpart="Sim1-positive V3 interneurons with dorsal/ventral intrinsic phenotypes and a ventrolateral connectivity subset",
        representation_kind="genetic_class_with_intrinsic_and_connectivity_subphenotypes",
        equation_id="E_V3_DORSOVENTRAL_IH_IT_VLAT_MOTOR_MICROCIRCUIT",
        equation_family="ventral fast-tonic AdEx / dorsal adapting AdEx + Ih/T-type Ca; connectivity-only V3_VLat motor microcircuit",
        transmitter="glutamate",
        capacitance_pf=162.0, leak_ns=9.4, adaptation_a_ns=0.95,
        adaptation_tau_ms=210.0, adaptation_b_pa=18.0, tonic_drive_pa=85.0,
        intrinsic_terms=(
            "I_LEAK", "I_ADEX_EXP", "I_ADAPT_W",
            "I_V3_H_VENTRAL", "I_V3_H_DORSAL", "I_V3_T_DORSAL",
        ),
        receptor_or_input_terms=(
            "RG_to_V3_glutamatergic_drive",
            "MN_to_V3_VLat_glutamatergic_recurrent_excitation",
        ),
        input_terms=("RG_drive_only_no_speed_offset", "MN_recurrent_glutamate_to_V3_VLat"),
        output_terms=(
            "V3_ventral_to_contralateral_flexor_and_extensor_MN",
            "V3_VLat_to_ipsilateral_MN",
        ),
        differentiating_domain=(
            "measured_intrinsic_subphenotypes_and_anatomically_supported_"
            "commissural_and_VLat_ipsilateral_motor_output"
        ),
        execution_contract_id="XC_V3",
        evidence_level=(
            "direct_intrinsic_and_VLat_ipsilateral_MN_recurrent_paths_"
            "crossed_MN_topology_is_indirect_and_exact_realization_is_H_level"
        ),
        literature_urls=(
            "https://doi.org/10.1523/JNEUROSCI.2005-13.2013",
            "https://doi.org/10.1016/j.neuron.2008.09.027",
            "https://doi.org/10.1016/j.celrep.2018.08.095",
            "https://doi.org/10.1016/j.celrep.2024.115212",
        ),
    ),
    "V1Ia": CellClassEquation(
        cell_class="V1Ia", biological_counterpart="V1-enriched reciprocal Ia inhibitory interneuron population",
        representation_kind="physiological_V1_enriched_subclass", equation_id="E_V1IA_AFFERENT_GLY_RECIPROCAL",
        equation_family="shared inhibitory AdEx core + explicit Ia-afferent state + reciprocal glycinergic output",
        transmitter="glycine_GABA",
        capacitance_pf=148.0, leak_ns=8.8, adaptation_a_ns=0.70,
        adaptation_tau_ms=170.0, adaptation_b_pa=14.0, tonic_drive_pa=20.0,
        intrinsic_terms=("I_LEAK", "I_ADEX_EXP", "I_ADAPT_W"),
        receptor_or_input_terms=("Ia_length_velocity_afferent", "Renshaw_inhibition", "V0V_crossed_excitation"),
        input_terms=("Ia_afferent", "RG_drive", "V0V_crossed_excitation", "Renshaw_inhibition"),
        output_terms=("antagonist_RG/PF/MN_glycinergic_inhibition",),
        differentiating_domain="proprioceptive_input_and_reciprocal_target",
        execution_contract_id="XC_V1IA",
        evidence_level="direct_circuit_identity_no_unique_channel_claim",
        literature_urls=(
            "https://doi.org/10.1152/jn.90354.2008",
            "https://doi.org/10.1113/jphysiol.2010.199125",
            "https://doi.org/10.7554/eLife.95172",
        ),
    ),
    "V1Ren": CellClassEquation(
        cell_class="V1Ren", biological_counterpart="Chrna2-positive V1-derived Renshaw cells",
        representation_kind="physiological_V1_subclass", equation_id="E_RENSHAW_MIXED_MN_INPUT_IH_SK",
        equation_family="AdEx + paired-topology effective MN-collateral nAChR/GluR components + Ih + SK",
        transmitter="glycine_GABA",
        capacitance_pf=125.0, leak_ns=10.5, adaptation_a_ns=0.20,
        adaptation_tau_ms=85.0, adaptation_b_pa=5.0, tonic_drive_pa=50.0,
        intrinsic_terms=(
            "I_LEAK", "I_ADEX_EXP", "I_ADAPT_W",
            "I_RENSHAW_H", "I_RENSHAW_SK",
        ),
        receptor_or_input_terms=(
            "MN_collateral_nAChR", "MN_collateral_glutamate_receptors",
        ),
        input_terms=("MN_paired_effective_ACh_glutamate_components",),
        output_terms=("Renshaw_to_MN_glycinergic", "Renshaw_to_IaIN_glycinergic"),
        differentiating_domain="intrinsic_receptor_and_recurrent_circuit",
        execution_contract_id="XC_V1REN",
        evidence_level="direct_mixed_input_and_intrinsic_mechanisms_receptor_mixtures_reduced",
        literature_urls=(
            "https://doi.org/10.1038/s41598-017-04266-8",
            "https://doi.org/10.1152/jn.90354.2008",
            "https://doi.org/10.1111/ejn.12852",
            "https://doi.org/10.1523/JNEUROSCI.2541-15.2015",
        ),
    ),
    "V2b": CellClassEquation(
        cell_class="V2b", biological_counterpart="Gata3-positive ipsilateral inhibitory V2b interneurons",
        representation_kind="genetic_class", equation_id="E_V2B_GLY_EXTENSOR_BIASED",
        equation_family="shared inhibitory AdEx core + ipsilateral extensor-biased glycinergic output",
        transmitter="glycine_GABA",
        capacitance_pf=148.0, leak_ns=8.8, adaptation_a_ns=0.70,
        adaptation_tau_ms=170.0, adaptation_b_pa=14.0, tonic_drive_pa=20.0,
        intrinsic_terms=("I_LEAK", "I_ADEX_EXP", "I_ADAPT_W"),
        receptor_or_input_terms=("RG_to_V2b_glutamatergic_drive",),
        input_terms=("RG_drive",),
        output_terms=("ipsilateral_extensor_biased_glycinergic_inhibition",),
        differentiating_domain="ipsilateral_target_bias_and_transmitter",
        execution_contract_id="XC_V2B",
        evidence_level="direct_identity_target_bias_no_unique_channel_claim",
        literature_urls=("https://doi.org/10.1016/j.neuron.2014.02.013", "https://doi.org/10.7554/eLife.04718"),
    ),
}


@dataclass(frozen=True)
class PhasePathwayContract:
    """Exact pathway names expected for one population in each phase."""

    common: Tuple[str, ...] = ()
    flexor_only: Tuple[str, ...] = ()
    extensor_only: Tuple[str, ...] = ()

    def for_phase(self, phase: str) -> Tuple[str, ...]:
        if phase not in PHASES:
            raise ValueError(f"Unknown phase in pathway contract: {phase}")
        return self.common + (
            self.flexor_only if phase == "F" else self.extensor_only
        )

    def all_names(self) -> Tuple[str, ...]:
        return self.common + self.flexor_only + self.extensor_only


@dataclass(frozen=True)
class ClassExecutionContract:
    """Typed, executable contract linking a class to currents and topology.

    This registry—not source-code substring matching—is the fail-closed bridge
    between a biological class label and the equations that are executed. A
    shared AdEx term set is permitted where class-specific channel evidence is
    absent; input, transmitter and exact projection topology must still differ.
    """

    cell_class: str
    contract_id: str
    transmitter: str
    output_weight_sign: int
    intrinsic_term_ids: Tuple[str, ...]
    direct_input_ids: Tuple[str, ...]
    peripheral_output_ids: Tuple[str, ...]
    incoming_pathways: PhasePathwayContract
    outgoing_pathways: PhasePathwayContract
    target_specific_transmitters: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ConnectivitySubphenotypeContract:
    """A measured connectivity subtype without an invented ion-current ID."""

    subphenotype: str
    parent_intrinsic_phenotype: str
    mask_symbol: str
    incoming_pathways: Tuple[str, ...]
    outgoing_pathways: Tuple[str, ...]
    source_ids: Tuple[str, ...]
    evidence_grade: str


COMMON_ADEX_RUNTIME_TERM_IDS = (
    "I_LEAK", "I_ADEX_EXP", "I_ADAPT_W",
)
RUNTIME_INTRINSIC_TERM_ORDER = COMMON_ADEX_RUNTIME_TERM_IDS + (
    "I_NAP_RG", "I_LTYPE_CA_PIC_RG", "I_M_RG", "I_KCA_RG",
    "I_H_RG", "I_T_RG", "I_A_RG", "I_RG_GAP", "I_PF_SLOW",
    "I_MN_DEND_LEAK", "I_MN_NAP_PIC", "I_MN_LTYPE_CA_PIC",
    "I_MN_COUPLING_SOMA", "I_MN_AHP",
    "I_V2A_H_TONIC", "I_V2A_H_PHASIC", "I_V2A_H_DELAYED",
    "I_V2A_DELAY", "I_V2A_GAP_TONIC", "I_V2A_GAP_PHASIC",
    "I_V3_H_VENTRAL", "I_V3_H_DORSAL",
    "I_V3_T_DORSAL",
    "I_RENSHAW_H", "I_RENSHAW_SK",
)
RUNTIME_INTRINSIC_TERM_IDS = frozenset(RUNTIME_INTRINSIC_TERM_ORDER)
DISABLABLE_INTRINSIC_TERM_IDS = frozenset(
    RUNTIME_INTRINSIC_TERM_IDS - set(COMMON_ADEX_RUNTIME_TERM_IDS)
)
RUNTIME_DIRECT_INPUT_TERM_ORDER = (
    "I_TONIC_CLASS", "I_DESCENDING_RG", "I_PERTURBATION",
    "I_IA_TO_PF_EFFECTIVE", "I_IA_TO_MN", "I_IA_TO_V1IA",
    "I_IB_TO_PF_EFFECTIVE", "I_IB_TO_MN_EFFECTIVE", "I_PF_DELETION",
)
DIRECT_INPUT_TERM_IDS = frozenset(RUNTIME_DIRECT_INPUT_TERM_ORDER)
PERIPHERAL_OUTPUT_TERM_IDS = frozenset(("E_NMJ_VESICLE_ACH",))


CLASS_EXECUTION_CONTRACTS: Dict[str, ClassExecutionContract] = {
    "RG": ClassExecutionContract(
        "RG", "XC_RG", "glutamate", 1,
        COMMON_ADEX_RUNTIME_TERM_IDS + (
            "I_NAP_RG", "I_LTYPE_CA_PIC_RG", "I_M_RG", "I_KCA_RG",
            "I_H_RG", "I_T_RG", "I_A_RG", "I_RG_GAP",
        ),
        ("I_TONIC_CLASS", "I_DESCENDING_RG", "I_PERTURBATION"), (),
        PhasePathwayContract(
            common=(
                "RG_recurrent", "V1Ia_to_antagonist_RG",
                "V2b_to_antagonist_RG", "V0D_cross_inhibition",
            ),
        ),
        PhasePathwayContract(common=(
            "RG_recurrent", "RG_to_V1Ia", "RG_to_V2b", "RG_to_PF",
            "RG_to_V2a", "RG_to_V0D", "RG_to_V3",
        )),
    ),
    "PF": ClassExecutionContract(
        "PF", "XC_PF", "glutamate", 1,
        COMMON_ADEX_RUNTIME_TERM_IDS + ("I_PF_SLOW",),
        (
            "I_TONIC_CLASS", "I_IA_TO_PF_EFFECTIVE",
            "I_IB_TO_PF_EFFECTIVE", "I_PF_DELETION",
        ), (),
        PhasePathwayContract(common=(
            "RG_to_PF", "PF_recurrent", "V1Ia_to_antagonist_PF",
            "V2b_to_antagonist_PF", "V2a_to_PF",
        )),
        PhasePathwayContract(common=("PF_recurrent", "PF_to_MN")),
    ),
    "MN": ClassExecutionContract(
        "MN", "XC_MN", "acetylcholine", 1,
        COMMON_ADEX_RUNTIME_TERM_IDS + (
            "I_MN_DEND_LEAK", "I_MN_NAP_PIC", "I_MN_LTYPE_CA_PIC",
            "I_MN_COUPLING_SOMA", "I_MN_AHP",
        ),
        ("I_TONIC_CLASS", "I_IA_TO_MN", "I_IB_TO_MN_EFFECTIVE"),
        ("E_NMJ_VESICLE_ACH",),
        PhasePathwayContract(common=(
            "PF_to_MN", "V1Ia_to_antagonist_MN", "V2b_to_antagonist_MN",
            "V1Ren_to_MN", "V2a_to_MN", "V3_VLat_to_ipsilateral_MN",
        ),
            flexor_only=("V3_ventral_to_contralateral_MN_flexor",),
            extensor_only=("V3_ventral_to_contralateral_MN_extensor",),
        ),
        PhasePathwayContract(common=(
            "MN_to_V1Ren_nAChR", "MN_to_V1Ren_GluR", "MN_to_V3_VLat_GluR",
        )),
        target_specific_transmitters=((
            "MN_to_V1Ren_nAChR", "acetylcholine",
        ), (
            "MN_to_V1Ren_GluR", "glutamate",
        ), (
            "MN_to_V3_VLat_GluR", "glutamate",
        )),
    ),
    "V0D": ClassExecutionContract(
        "V0D", "XC_V0D", "glycine_GABA", -1,
        COMMON_ADEX_RUNTIME_TERM_IDS, ("I_TONIC_CLASS",), (),
        PhasePathwayContract(common=("RG_to_V0D",)),
        PhasePathwayContract(common=("V0D_cross_inhibition",)),
    ),
    "V0V": ClassExecutionContract(
        "V0V", "XC_V0V", "glutamate", 1,
        COMMON_ADEX_RUNTIME_TERM_IDS, ("I_TONIC_CLASS",), (),
        PhasePathwayContract(common=("V2a_to_V0V",)),
        PhasePathwayContract(common=("V0V_to_cross_V1Ia",)),
    ),
    "V2a": ClassExecutionContract(
        "V2a", "XC_V2A", "glutamate", 1,
        COMMON_ADEX_RUNTIME_TERM_IDS + (
            "I_V2A_H_TONIC", "I_V2A_H_PHASIC", "I_V2A_H_DELAYED",
            "I_V2A_DELAY", "I_V2A_GAP_TONIC", "I_V2A_GAP_PHASIC",
        ),
        ("I_TONIC_CLASS",), (),
        PhasePathwayContract(common=("RG_to_V2a",)),
        PhasePathwayContract(common=("V2a_to_V0V", "V2a_to_PF", "V2a_to_MN")),
    ),
    "V3": ClassExecutionContract(
        "V3", "XC_V3", "glutamate", 1,
        COMMON_ADEX_RUNTIME_TERM_IDS + (
            "I_V3_H_VENTRAL", "I_V3_H_DORSAL", "I_V3_T_DORSAL",
        ),
        ("I_TONIC_CLASS",), (),
        PhasePathwayContract(common=("RG_to_V3", "MN_to_V3_VLat_GluR")),
        PhasePathwayContract(
            common=("V3_VLat_to_ipsilateral_MN",),
            flexor_only=("V3_ventral_to_contralateral_MN_flexor",),
            extensor_only=("V3_ventral_to_contralateral_MN_extensor",),
        ),
    ),
    "V1Ia": ClassExecutionContract(
        "V1Ia", "XC_V1IA", "glycine_GABA", -1,
        COMMON_ADEX_RUNTIME_TERM_IDS,
        ("I_TONIC_CLASS", "I_IA_TO_V1IA"), (),
        PhasePathwayContract(common=(
            "RG_to_V1Ia", "V0V_to_cross_V1Ia", "V1Ren_to_V1Ia",
        )),
        PhasePathwayContract(common=(
            "V1Ia_to_antagonist_RG", "V1Ia_to_antagonist_PF",
            "V1Ia_to_antagonist_MN",
        )),
    ),
    "V1Ren": ClassExecutionContract(
        "V1Ren", "XC_V1REN", "glycine_GABA", -1,
        COMMON_ADEX_RUNTIME_TERM_IDS + ("I_RENSHAW_H", "I_RENSHAW_SK"),
        ("I_TONIC_CLASS",), (),
        PhasePathwayContract(common=(
            "MN_to_V1Ren_nAChR", "MN_to_V1Ren_GluR",
        )),
        PhasePathwayContract(common=("V1Ren_to_MN", "V1Ren_to_V1Ia")),
    ),
    "V2b": ClassExecutionContract(
        "V2b", "XC_V2B", "glycine_GABA", -1,
        COMMON_ADEX_RUNTIME_TERM_IDS, ("I_TONIC_CLASS",), (),
        PhasePathwayContract(common=("RG_to_V2b",)),
        PhasePathwayContract(common=(
            "V2b_to_antagonist_RG", "V2b_to_antagonist_PF",
            "V2b_to_antagonist_MN",
        )),
    ),
}


V3_CONNECTIVITY_SUBPHENOTYPE_CONTRACTS: Dict[
    str, ConnectivitySubphenotypeContract
] = {
    "V3_VLat": ConnectivitySubphenotypeContract(
        subphenotype="V3_VLat",
        parent_intrinsic_phenotype="V3_ventral",
        mask_symbol="v3_vlat_connectivity_mask",
        incoming_pathways=("MN_to_V3_VLat_GluR",),
        outgoing_pathways=("V3_VLat_to_ipsilateral_MN",),
        source_ids=("CHOPEK2018_V3_MICROCIRCUIT",),
        evidence_grade="A",
    ),
}


def validate_class_execution_contracts() -> None:
    """Reject label-only, unknown-term or metadata/contract drift."""
    if set(CLASS_EXECUTION_CONTRACTS) != set(CLASSES):
        raise ValueError("Class execution-contract registry must exactly cover CLASSES")
    contracts = tuple(CLASS_EXECUTION_CONTRACTS.values())
    if len({record.contract_id for record in contracts}) != len(contracts):
        raise ValueError("Every class execution contract must have a unique id")
    for key, contract in CLASS_EXECUTION_CONTRACTS.items():
        if key != contract.cell_class:
            raise ValueError(f"Class execution-contract key mismatch: {key}")
        equation = CELL_CLASS_EQUATIONS[key]
        if equation.execution_contract_id != contract.contract_id:
            raise ValueError(f"{key} equation/typed execution-contract id mismatch")
        if equation.intrinsic_terms != contract.intrinsic_term_ids:
            raise ValueError(f"{key} equation/runtime intrinsic-term ids drifted")
        if equation.transmitter != contract.transmitter:
            raise ValueError(f"{key} transmitter differs between registries")
        if (
            equation.target_specific_transmitters
            != contract.target_specific_transmitters
        ):
            raise ValueError(
                f"{key} target-specific transmitter contract drifted"
            )
        if contract.output_weight_sign not in {-1, 1}:
            raise ValueError(f"{key} has invalid output sign")
        expected_sign = -1 if contract.transmitter == "glycine_GABA" else 1
        if contract.output_weight_sign != expected_sign:
            raise ValueError(f"{key} transmitter/output sign mismatch")
        if not contract.intrinsic_term_ids:
            raise ValueError(f"{key} has no executable intrinsic terms")
        if not set(contract.intrinsic_term_ids) <= RUNTIME_INTRINSIC_TERM_IDS:
            raise ValueError(f"{key} declares an unknown intrinsic runtime term")
        if len(set(contract.intrinsic_term_ids)) != len(contract.intrinsic_term_ids):
            raise ValueError(f"{key} repeats an intrinsic runtime term")
        if not set(contract.direct_input_ids) <= DIRECT_INPUT_TERM_IDS:
            raise ValueError(f"{key} declares an unknown direct-input term")
        if not set(contract.peripheral_output_ids) <= PERIPHERAL_OUTPUT_TERM_IDS:
            raise ValueError(f"{key} declares an unknown peripheral output")
        if len(set(contract.direct_input_ids)) != len(contract.direct_input_ids):
            raise ValueError(f"{key} repeats a direct-input term")
        if len(set(contract.peripheral_output_ids)) != len(
            contract.peripheral_output_ids
        ):
            raise ValueError(f"{key} repeats a peripheral output")
        for direction in (contract.incoming_pathways, contract.outgoing_pathways):
            names = direction.all_names()
            if not names or len(names) != len(set(names)):
                raise ValueError(f"{key} has empty or duplicate pathway declarations")
        transmitter_overrides = dict(contract.target_specific_transmitters)
        if len(transmitter_overrides) != len(
            contract.target_specific_transmitters
        ):
            raise ValueError(f"{key} repeats a target-specific transmitter")
        if not set(transmitter_overrides) <= set(
            contract.outgoing_pathways.all_names()
        ):
            raise ValueError(
                f"{key} transmitter override references a non-outgoing pathway"
            )
        if not set(transmitter_overrides.values()) <= {
            "glutamate", "glycine_GABA", "acetylcholine",
            "acetylcholine_glutamate",
        }:
            raise ValueError(f"{key} has an unsupported target transmitter")
        expected_overrides = (
            {
                "MN_to_V1Ren_nAChR": "acetylcholine",
                "MN_to_V1Ren_GluR": "glutamate",
                "MN_to_V3_VLat_GluR": "glutamate",
            }
            if key == "MN" else {}
        )
        if transmitter_overrides != expected_overrides:
            raise ValueError(
                f"{key} target-specific transmitter identity drifted"
            )


def validate_cell_class_equations() -> None:
    """Enforce the no-label-without-executable-mechanism-and-evidence rule."""
    if set(CELL_CLASS_EQUATIONS) != set(CLASSES):
        missing = set(CLASSES) - set(CELL_CLASS_EQUATIONS)
        extra = set(CELL_CLASS_EQUATIONS) - set(CLASSES)
        raise ValueError(f"Cell-class equation registry mismatch: missing={missing}, extra={extra}")
    records = tuple(CELL_CLASS_EQUATIONS.values())
    if len({record.equation_id for record in records}) != len(records):
        raise ValueError("Every cell class must have a unique equation_id")
    if len({record.mechanistic_signature() for record in records}) != len(records):
        raise ValueError("Every class must have a distinct full mechanistic signature")
    forbidden_model_only_subtypes = {"V2aComm", "V2aIpsi", "V1Spd", "V2bRec", "V2bPre"}
    if forbidden_model_only_subtypes & set(CLASSES):
        raise ValueError("Unsupported model-only subtype retained")
    for record in records:
        if not record.biological_counterpart or not record.intrinsic_terms:
            raise ValueError(f"{record.cell_class} lacks biological/equation counterpart")
        if not record.receptor_or_input_terms or not record.differentiating_domain:
            raise ValueError(f"{record.cell_class} lacks an operative differentiation domain")
        if not record.input_terms or not record.output_terms:
            raise ValueError(f"{record.cell_class} lacks an explicit input/output role")
        if not record.execution_contract_id:
            raise ValueError(f"{record.cell_class} lacks a typed execution contract")
        for parameter_name in (
            "capacitance_pf", "leak_ns", "adaptation_tau_ms",
        ):
            value = getattr(record, parameter_name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{record.cell_class}.{parameter_name} must be finite and positive"
                )
        for parameter_name in ("adaptation_a_ns", "adaptation_b_pa"):
            value = getattr(record, parameter_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{record.cell_class}.{parameter_name} must be finite and nonnegative"
                )
        if not math.isfinite(record.tonic_drive_pa):
            raise ValueError(f"{record.cell_class}.tonic_drive_pa must be finite")
        if record.transmitter not in {"glutamate", "glycine_GABA", "acetylcholine"}:
            raise ValueError(f"{record.cell_class} has an unsupported transmitter")
        if not record.literature_urls or not all(url.startswith("https://") for url in record.literature_urls):
            raise ValueError(f"{record.cell_class} lacks literature traceability")
    validate_class_execution_contracts()


@dataclass(frozen=True)
class BiologicalInterfaceEquation:
    """Contract for every named biological interface outside cell identity."""

    interface: str
    biological_counterpart: str
    equation_id: str
    state_variables: Tuple[str, ...]
    equation_terms: Tuple[str, ...]
    input_role: str
    output_role: str
    evidence_level: str
    literature_urls: Tuple[str, ...]
    implementation_symbols: Tuple[str, ...]
    parameter_status: str = "reduced_model_prior_not_cross_species_point_estimate"


BIOLOGICAL_INTERFACE_EQUATIONS: Dict[str, BiologicalInterfaceEquation] = {
    "central_chemical_synapse": BiologicalInterfaceEquation(
        "central_chemical_synapse", "delayed chemical synaptic transmission",
        "E_CNS_SYN_DELAY_CONDUCTANCE", (
            "scheduled_exc_conductance", "scheduled_inh_conductance",
            "syn_exc_conductance", "syn_inh_conductance",
        ),
        ("delayed_release", "separate_receptor_decay", "reversal_potential_current"),
        "presynaptic spikes and transmitter identity", "glutamatergic or glycine/GABA conductance current",
        "computational_kinetic_provenance_not_spinal_receptor_identity",
        ("https://doi.org/10.1162/neco.1994.6.1.14",),
        ("scheduled_exc_conductance", "scheduled_inh_conductance", "i_syn_exc", "i_syn_inh"),
    ),
    "renshaw_mixed_mn_input": BiologicalInterfaceEquation(
        "renshaw_mixed_mn_input",
        "segregated effective cholinergic and glutamatergic MN-to-Renshaw input",
        "E_RENSHAW_MIXED_COTRANSMISSION",
        ("renshaw_nachr_conductance", "renshaw_glutamate_conductance"),
        (
            "MN_to_V1Ren_nAChR", "MN_to_V1Ren_GluR",
            "shared_cell_pair_topology_separate_effective_component_states",
            "separate_effective_receptor_decay",
            "separate_reversal_potential_currents",
        ),
        "motoneuron collateral spikes",
        "effective nAChR and glutamate-receptor currents in V1Ren",
        "direct_single_pair_mixed_input_and_transmitter_system_segregation_supported_component_edge_resource_delay_receptor_kinetics_and_equal_gain_are_H_level",
        ("https://doi.org/10.1038/s41598-017-04266-8",),
        (
            "scheduled_nachr_conductance", "scheduled_renshaw_glutamate_conductance",
            "renshaw_nachr_conductance", "renshaw_glutamate_conductance",
            "i_renshaw_nachr", "i_renshaw_glutamate",
        ),
    ),
    "nmj_presynaptic_release": BiologicalInterfaceEquation(
        "nmj_presynaptic_release", "alpha-MN terminal vesicle/ACh release",
        "E_NMJ_VESICLE_ACH", (
            "scheduled_nmj_event_fraction", "nmj_vesicle_available",
            "nmj_ach_gate",
        ),
        (
            "fixed_physical_release_latency", "vesicle_recovery",
            "activity_dependent_release", "ACh_clearance",
        ),
        "terminal-arrived MN population spike events", "effective synaptic-cleft ACh gate",
        "mechanisms_supported_single_pool_rates_and_gain_are_H_level",
        (
            "https://doi.org/10.1113/jphysiol.1972.sp010054",
            "https://doi.org/10.1111/j.1471-4159.2006.04282.x",
            "https://doi.org/10.1016/S0896-6273(03)00405-7",
            "https://doi.org/10.1111/bph.15940",
        ),
        (
            "scheduled_nmj_event_fraction", "nmj_vesicle_available",
            "nmj_released", "nmj_ach_gate",
        ),
    ),
    "nmj_postsynaptic_endplate": BiologicalInterfaceEquation(
        "nmj_postsynaptic_endplate", "nicotinic end-plate depolarization",
        "E_NMJ_NACHR_EPP", ("nmj_nachr_open", "nmj_endplate_mv", "muscle_fiber_excitation"),
        ("ACh_nAChR_binding", "endplate_depolarization", "fiber_threshold"),
        "cleft ACh", "nAChR opening, end-plate potential and fiber excitation",
        "endplate_channel_supported_Hill_binding_sigmoid_fiber_threshold_and_gains_are_H_level",
        ("https://doi.org/10.1113/jphysiol.1973.sp010410",),
        ("nmj_endplate_mv", "nmj_nachr_open", "muscle_fiber_excitation"),
    ),
    "excitation_contraction": BiologicalInterfaceEquation(
        "excitation_contraction", "muscle-fiber Ca2+ and contractile activation",
        "E_MUSCLE_EFFECTIVE_SR_CA_ACTIVATION", ("muscle_calcium", "muscle_activation"),
        ("effective_SR_Ca_release_clearance", "saturating_Ca_activation"),
        "muscle-fiber excitation", "contractile activation",
        "mechanism_supported_gain_is_model_prior",
        (
            "https://doi.org/10.1038/2191168a0",
            "https://doi.org/10.1038/325717a0",
            "https://doi.org/10.1085/jgp.111.4.505",
        ),
        ("muscle_calcium", "calcium_activation", "muscle_activation"),
    ),
    "muscle_force": BiologicalInterfaceEquation(
        "muscle_force", "antagonistic muscle force generation",
        "E_MUSCLE_ACTIVE_FORCE", ("muscle_force",),
        ("activation_force", "extensor_force_scale_prior"),
        "contractile activation", "normalized flexor/extensor active force",
        "phenomenological_normalized_force_gain_extensor_scale_is_H_level",
        ("https://doi.org/10.1085/jgp.111.4.505",),
        ("muscle_force", "extensor_force_scale_prior", "muscle_activation"),
    ),
    "joint_mechanics": BiologicalInterfaceEquation(
        "joint_mechanics", "single-degree antagonistic limb segment",
        "E_JOINT_INERTIA_DAMPING_STIFFNESS", ("joint_position", "joint_velocity"),
        ("inertia", "damping", "external_resistive_load", "stiffness", "antagonistic_torque"),
        "muscle forces and external resistive load", "position and velocity",
        "inertia_damping_stiffness_supported_external_resistance_multiplier_is_H_level",
        ("https://doi.org/10.1016/0021-9290(82)90089-6",),
        ("joint_position", "joint_velocity", "torque", "load_resistance_factors"),
    ),
    "Ia_spindle_transducer": BiologicalInterfaceEquation(
        "Ia_spindle_transducer", "muscle-spindle primary length/velocity proxy",
        "E_IA_SPINDLE_LENGTH_VELOCITY_PROXY", ("latest_ia_signal",),
        ("length_component", "velocity_component", "bounded_normalization"),
        "muscle length and lengthening velocity", "bounded Ia-like spindle proxy",
        "primary_sensor_reduction_not_a_complete_spindle_model",
        ("https://doi.org/10.1152/jn.00868.2005",),
        ("latest_ia_signal", "muscle_length", "muscle_length_velocity"),
    ),
    "Ia_effective_spinal_pathway": BiologicalInterfaceEquation(
        "Ia_effective_spinal_pathway",
        "effective spinal pathways downstream of the Ia-like spindle proxy",
        "E_IA_EFFECTIVE_SPINAL_CURRENT",
        ("latest_ia_transmission", "sensory_resource"),
        (
            "phenomenological_use_dependent_resource",
            "effective_PF_current", "effective_MN_current",
            "effective_V1Ia_current",
        ),
        "bounded Ia-like spindle proxy",
        "effective PF/MN/V1Ia current in the represented locomotor circuit",
        "mixed_direct_circuit_support_PF_and_gains_are_H_level",
        ("https://doi.org/10.1152/jn.90354.2008",),
        (
            "latest_ia_transmission", "sensory_resource",
            "ia_to_pf_pa", "ia_to_mn_pa", "ia_to_v1ia_pa",
        ),
    ),
    "Ib_GTO_transducer": BiologicalInterfaceEquation(
        "Ib_GTO_transducer", "Golgi-tendon-organ normalized force proxy",
        "E_IB_GTO_FORCE_PROXY", ("latest_ib_signal",),
        ("tonic_force_transduction", "bounded_normalization"),
        "normalized active muscle force", "bounded Ib-like force proxy",
        "primary_sensor_reduction_not_a_central_reflex_sign",
        ("https://doi.org/10.1152/jn.00869.2005",),
        ("latest_ib_signal", "ib_force_gain", "muscle_force"),
    ),
    "Ib_effective_spinal_pathway": BiologicalInterfaceEquation(
        "Ib_effective_spinal_pathway",
        "unresolved effective locomotor spinal pathway downstream of the Ib-like proxy",
        "E_IB_EFFECTIVE_SPINAL_CURRENT",
        ("latest_ib_transmission", "sensory_resource"),
        (
            "phenomenological_use_dependent_resource",
            "effective_PF_current", "effective_MN_current",
            "represented_extensor_context_prior",
        ),
        "bounded Ib-like force proxy",
        "effective PF/MN current in the represented locomotor regime",
        "H_level_effective_path_not_monosynaptic_Ib_excitation",
        (
            "https://doi.org/10.1113/jphysiol.1983.sp014663",
            "https://doi.org/10.1152/jn.1993.70.3.1009",
            "https://doi.org/10.1007/BF00228410",
        ),
        (
            "latest_ib_transmission", "sensory_resource",
            "ib_effective_spinal_to_pf_pa",
            "ib_effective_spinal_to_mn_pa",
            "ib_effective_extensor_context_gain",
        ),
    ),
    "local_mt_terminal": BiologicalInterfaceEquation(
        "local_mt_terminal", "activity-nucleated presynaptic microtubule tracks and slow vesicle replenishment",
        "E_MT_LOCAL_TRACK_SLOW_REPLENISHMENT",
        ("mt_edge_activity", "mt_edge_tracks", "edge_slow_replenishment_resource", "edge_rrp_available"),
        ("activity_dependent_nucleation", "track_turnover", "MT_dependent_slow_replenishment", "MT_independent_fast_replenishment", "spike_depletion"),
        "local presynaptic spikes", "normalized-resource-gated slow RRP replenishment and subsequent release availability",
        "two_nonspinal_domains_combined_all_modeled_central_chemical_routes_and_normalized_resource_law_are_H_level",
        (
            "https://doi.org/10.1016/j.cub.2019.10.049",
            "https://doi.org/10.1523/JNEUROSCI.1571-19.2019",
        ),
        ("mt_edge_activity", "mt_edge_tracks", "edge_slow_replenishment_resource", "edge_rrp_available"),
    ),
}


@dataclass(frozen=True)
class LiteratureSource:
    """Canonical primary/computational source with an explicit domain limit."""

    source_id: str
    title: str
    url: str
    evidence_domain: str
    preparation: str
    scope_limit: str


@dataclass(frozen=True)
class EvidenceBinding:
    """Fail-closed source binding for one executable biological claim.

    Grades are: A, direct experimental mechanism/circuit evidence; B,
    genetic-functional identity or supported motif; C, original computational
    provenance; H, exact model hypothesis. A/B/C never certify the numerical
    implementation: ``h_level_boundary`` states what remains a model prior.
    """

    mechanism_evidence: str
    realization_evidence: str
    source_ids: Tuple[str, ...]
    supported_scope: str
    h_level_boundary: str


LITERATURE_SOURCE_TITLES: Dict[str, str] = {
    "BRETTE_GERSTNER2005_ADEX": "Adaptive Exponential Integrate-and-Fire Model as an Effective Description of Neuronal Activity",
    "DOUGHERTY2013_SHOX2_RHYTHM": "Locomotor Rhythm Generation Linked to the Output of Spinal Shox2 Excitatory Interneurons",
    "SINGH2025_SHOX2_CURRENTS": "Properties of rhythmogenic currents in spinal Shox2 interneurons across postnatal development",
    "HA_DOUGHERTY2018_SHOX2_COUPLING": "Spinal Shox2 interneuron interconnectivity related to function and development",
    "LAFRENIERE_ROULA2005_CPG_DELETIONS": "Deletions of Rhythmic Motoneuron Activity During Fictive Locomotion and Scratch Provide Clues to the Organization of the Mammalian Central Pattern Generator",
    "ZHONG2012_MOUSE_CPG_DELETIONS": "Neuronal activity in the isolated mouse spinal cord during spontaneous deletions in fictive locomotion: insights into locomotor central pattern generator organization",
    "CARLIN2000_MN_DEND_LCA": "Dendritic L-type calcium currents in mouse spinal motoneurons: implications for bistability",
    "LI_BENNETT2003_MN_NAP_LCA_PIC": "Persistent Sodium and Calcium Currents Cause Plateau Potentials in Motoneurons of Chronic Spinal Rats",
    "SCHWINDT_CRILL1980_MN_PIC": "Properties of a persistent inward current in normal and TEA-injected motoneurons",
    "LI_BENNETT2007_MN_SK": "Apamin-Sensitive Calcium-Activated Potassium Currents (SK) Are Activated by Persistent Calcium Currents in Rat Motoneurons",
    "LANUZA2004_V0_COORDINATION": "Genetic Identification of Spinal Interneurons that Coordinate Left-Right Locomotor Activity Necessary for Walking Movements",
    "TALPALAR2013_V0_SPEED": "Dual-mode operation of neuronal networks involved in left-right alternation",
    "MORAN_RIVARD2001_V0V_IDENTITY": "Evx1 Is a Postmitotic Determinant of V0 Interneuron Identity in the Spinal Cord",
    "CRONE2008_V2A_V0V": "Genetic Ablation of V2a Ipsilateral Interneurons Disrupts Left-Right Locomotor Coordination in Mammalian Spinal Cord",
    "ZHONG2010_V2A_PHENOTYPES": "Electrophysiological Characterization of V2a Interneurons and Their Locomotor-Related Activity in the Neonatal Mouse Spinal Cord",
    "HAYASHI2018_V2A_ARRAYS": "Graded Arrays of Spinal and Supraspinal V2a Interneuron Subtypes Underlie Forelimb and Hindlimb Motor Control",
    "BOROWSKA2013_V3_SUBPOPS": "Functional Subpopulations of V3 Interneurons in the Mature Mouse Spinal Cord",
    "ZHANG2008_V3_BALANCE": "V3 Spinal Neurons Establish a Robust and Balanced Locomotor Rhythm during Walking",
    "CHOPEK2018_V3_MICROCIRCUIT": "Sub-populations of Spinal V3 Interneurons Form Focal Modules of Layered Pre-motor Microcircuits",
    "ZHANG2025_V3_MOTOR_GAIN": "Widespread innervation of motoneurons by spinal V3 neurons globally amplifies locomotor output in mice",
    "WANG2008_IA_RECIPROCAL": "Early Postnatal Development of Reciprocal Ia Inhibition in the Murine Spinal Cord",
    "GEERTSEN2011_IA_LOCOMOTION": "Reciprocal Ia inhibition contributes to motoneuronal hyperpolarisation during the inactive phase of locomotion and scratching in the cat",
    "WORTHY2024_V1_CLADES": "Spinal V1 inhibitory interneuron clades differ in birthdate, projections to motoneurons, and heterogeneity",
    "LAMOTTE_DINCAMPS2017_MN_RC_MIXED": "Segregation of glutamatergic and cholinergic transmission at the mixed motoneuron Renshaw cell synapse",
    "PERRY2015_RENSHAW_IH_SK": "Firing properties of Renshaw cells defined by Chrna2 are modulated by hyperpolarizing and small conductance ion currents Ih and ISK",
    "MOORE2015_RENSHAW_RECURRENT": "Synaptic Connectivity between Renshaw Cells and Motoneurons in the Recurrent Inhibitory Circuit of the Spinal Cord",
    "ZHANG2014_V1_V2B_ALTERNATION": "V1 and V2b Interneurons Secure the Alternating Flexor-Extensor Motor Activity Mice Require for Limbed Locomotion",
    "BRITZ2015_V1_V2B_ASYMMETRY": "A genetically defined asymmetry underlies the inhibitory control of flexor-extensor locomotor movements",
    "DESTEXHE1994_SYN_KINETIC": "An Efficient Method for Computing Synaptic Conductances Based on a Kinetic Model of Receptor Binding",
    "BARRETT_STEVENS1972_NMJ_RELEASE": "The kinetics of transmitter release at the frog neuromuscular junction",
    "BUKHARAEVA2007_NMJ_RELEASE_SYNC": "Modulation of the kinetics of evoked quantal release at mouse neuromuscular junctions by calcium and strontium",
    "RICHARDS2003_NMJ_VESICLE_POOLS": "Synaptic Vesicle Pools at the Frog Neuromuscular Junction",
    "REDMAN2022_NMJ_ACHE": "Donepezil inhibits neuromuscular junctional acetylcholinesterase and enhances synaptic transmission and function in isolated skeletal muscle",
    "ANDERSON_STEVENS1973_NMJ_CHANNEL": "Voltage clamp analysis of acetylcholine produced end-plate current fluctuations at frog neuromuscular junction",
    "ASHLEY_RIDGWAY1968_MUSCLE_CA_TENSION": "Simultaneous Recording of Membrane Potential, Calcium Transient and Tension in Single Muscle Fibres",
    "RIOS_BRUM1987_EC_COUPLING": "Involvement of dihydropyridine receptors in excitation-contraction coupling in skeletal muscle",
    "KONISHI_WATANABE1998_CA_FORCE": "Steady State Relation between Cytoplasmic Free Ca2+ Concentration and Force in Intact Frog Skeletal Muscle Fibers",
    "HUNTER_KEARNEY1982_ANKLE_DYNAMICS": "Dynamics of human ankle stiffness: Variation with mean ankle torque",
    "MILEUSNIC2006_SPINDLE_MODEL": "Mathematical Models of Proprioceptors. I. Control and Transduction in the Muscle Spindle",
    "MILEUSNIC_LOEB2006_GTO_MODEL": "Mathematical Models of Proprioceptors. II. Structure and Function of the Golgi Tendon Organ",
    "JANKOWSKA_MCCREA1983_IB_PATHS": "Shared reflex pathways from Ib tendon organ afferents and Ia muscle spindle afferents in the cat.",
    "PEARSON_COLLINS1993_IB_REVERSAL": "Reversal of the influence of group Ib afferents from plantaris on activity in medial gastrocnemius muscle during locomotor activity",
    "GOSSARD1994_IB_LOCOMOTOR": "Transmission in a locomotor-related group Ib pathway from hindlimb extensor muscles in the cat",
    "QU2019_PRESYN_MT": "Activity-Dependent Nucleation of Dynamic Microtubules at Presynaptic Boutons Controls Neurotransmission",
    "BABU2020_MT_RECOVERY": "Microtubule and Actin Differentially Regulate Synaptic Vesicle Cycling to Maintain High-Frequency Neurotransmission",
}


def _source(
    source_id: str,
    doi: str,
    evidence_domain: str,
    preparation: str,
    scope_limit: str,
) -> LiteratureSource:
    if source_id not in LITERATURE_SOURCE_TITLES:
        raise RuntimeError(f"Missing canonical title for {source_id}")
    return LiteratureSource(
        source_id, LITERATURE_SOURCE_TITLES[source_id],
        f"https://doi.org/{doi}", evidence_domain,
        preparation, scope_limit,
    )


LITERATURE_SOURCES: Dict[str, LiteratureSource] = {
    "BRETTE_GERSTNER2005_ADEX": _source(
        "BRETTE_GERSTNER2005_ADEX", "10.1152/jn.00686.2005",
        "computational_neuron_equation", "generic fitted spike trains",
        "AdEx provenance; not spinal cell-class channel evidence",
    ),
    "DOUGHERTY2013_SHOX2_RHYTHM": _source(
        "DOUGHERTY2013_SHOX2_RHYTHM", "10.1016/j.neuron.2013.08.015",
        "Shox2 identity and locomotor-rhythm contribution", "neonatal mouse spinal cord",
        "does not establish every exact RG target or channel combination",
    ),
    "SINGH2025_SHOX2_CURRENTS": _source(
        "SINGH2025_SHOX2_CURRENTS", "10.1113/JP287752",
        "Shox2 intrinsic-current prevalence and pharmacology", "mouse P2-5, P14-21 and P60+ slices",
        "adult NaP/L-Ca pharmacology and sample prevalences do not fix model gates/gains or coexpression",
    ),
    "HA_DOUGHERTY2018_SHOX2_COUPLING": _source(
        "HA_DOUGHERTY2018_SHOX2_COUPLING", "10.7554/eLife.42519",
        "within-Shox2 electrical coupling", "mouse P0-5 paired recordings plus older ages",
        "nearby/process-selected pair incidence does not specify the model's sparse graph realization or gain",
    ),
    "LAFRENIERE_ROULA2005_CPG_DELETIONS": _source(
        "LAFRENIERE_ROULA2005_CPG_DELETIONS", "10.1152/jn.00216.2005",
        "rhythm-pattern functional separation", "adult decerebrate cat fictive locomotion/scratch",
        "does not identify a molecular PF class or PF-specific slow current",
    ),
    "ZHONG2012_MOUSE_CPG_DELETIONS": _source(
        "ZHONG2012_MOUSE_CPG_DELETIONS", "10.1113/jphysiol.2012.240895",
        "rhythm-pattern functional separation", "isolated mouse spinal cord",
        "does not identify the modeled PF state, targets or gains",
    ),
    "CARLIN2000_MN_DEND_LCA": _source(
        "CARLIN2000_MN_DEND_LCA", "10.1046/j.1460-9568.2000.00055.x",
        "motoneuron dendritic L-type Ca current", "mouse spinal motoneurons",
        "does not determine the reduced two-compartment parameters",
    ),
    "LI_BENNETT2003_MN_NAP_LCA_PIC": _source(
        "LI_BENNETT2003_MN_NAP_LCA_PIC", "10.1152/jn.00236.2003",
        "separate motoneuron Na and L-type Ca PICs", "chronic spinal rat motoneurons",
        "injury/species transfer and exact kinetics remain model priors",
    ),
    "SCHWINDT_CRILL1980_MN_PIC": _source(
        "SCHWINDT_CRILL1980_MN_PIC", "10.1152/jn.1980.43.6.1700",
        "motoneuron persistent inward current", "cat motoneurons",
        "PIC evidence, not SK/AHP evidence or model parameter values",
    ),
    "LI_BENNETT2007_MN_SK": _source(
        "LI_BENNETT2007_MN_SK", "10.1152/jn.01068.2006",
        "apamin-sensitive motoneuron SK/mAHP", "rat motoneurons",
        "does not specify the normalized Ca state or model conductance",
    ),
    "LANUZA2004_V0_COORDINATION": _source(
        "LANUZA2004_V0_COORDINATION", "10.1016/S0896-6273(04)00249-1",
        "V0 commissural identity and left-right coordination", "mouse spinal cord genetics",
        "broad inhibitory/excitatory V0 motif, not exact modeled RG targets",
    ),
    "TALPALAR2013_V0_SPEED": _source(
        "TALPALAR2013_V0_SPEED", "10.1038/nature12286",
        "speed-dependent V0 commissural roles", "mouse locomotor preparations",
        "does not specify exact model gains, delays or V1Ia target",
    ),
    "MORAN_RIVARD2001_V0V_IDENTITY": _source(
        "MORAN_RIVARD2001_V0V_IDENTITY", "10.1016/S0896-6273(01)00213-6",
        "Evx1/V0V identity", "mouse spinal development",
        "identity evidence does not establish the modeled output topology",
    ),
    "CRONE2008_V2A_V0V": _source(
        "CRONE2008_V2A_V0V", "10.1016/j.neuron.2008.08.009",
        "V2a contribution to high-speed left-right coordination", "mouse spinal locomotion",
        "supports the V2a-V0V motif, not every exact weight/delay",
    ),
    "ZHONG2010_V2A_PHENOTYPES": _source(
        "ZHONG2010_V2A_PHENOTYPES", "10.1523/JNEUROSCI.4849-09.2010",
        "V2a firing phenotypes, sag/rebound and electrical coupling", "P2-4 mouse Chx10 neurons",
        "Ih is current-clamp-consistent; exact gates, deterministic masks and delayed-current identity are H-level",
    ),
    "HAYASHI2018_V2A_ARRAYS": _source(
        "HAYASHI2018_V2A_ARRAYS", "10.1016/j.neuron.2018.01.023",
        "V2a spinal/supraspinal organization and ipsilateral motoneuron output",
        "mouse developmental genetics, tracing, synaptic-contact anatomy, rabies and photostimulation",
        "supports the population-level V2a-to-MN motif, not a mapping from tonic/phasic/delayed firing phenotypes to Type-I/II or transcriptomic subtypes, nor model gains/delays",
    ),
    "BOROWSKA2013_V3_SUBPOPS": _source(
        "BOROWSKA2013_V3_SUBPOPS", "10.1523/JNEUROSCI.2005-13.2013",
        "V3 dorsoventral physiology, Ih and dorsal T-type rebound", "juvenile P20-23 mouse T13-L3 slices; recordings P21-23",
        "channel subtypes, proportions and exact model kinetics/gains remain priors",
    ),
    "ZHANG2008_V3_BALANCE": _source(
        "ZHANG2008_V3_BALANCE", "10.1016/j.neuron.2008.09.027",
        "V3 identity, bilateral locomotor balance and predominantly contralateral flexor/extensor motor-pool input",
        "neonatal mouse genetics, PRV motor-pool tracing and spinal locomotion",
        "supports anatomical/transsynaptic motor-pool targeting but not paired functional EPSCs, model probabilities, delays, currents or online phase assignment",
    ),
    "CHOPEK2018_V3_MICROCIRCUIT": _source(
        "CHOPEK2018_V3_MICROCIRCUIT", "10.1016/j.celrep.2018.08.095",
        "ventral-V3 layered premotor microcircuit, V3-VLat-to-ipsilateral-MN output and recurrent glutamatergic MN-to-ventral-V3 input",
        "P7-14 mouse L1/L2 slices, patch recording and holographic glutamate uncaging",
        "supports the named circuit motifs and glutamatergic sign; V3-VLat fraction, phase/pool assignment, probability, delay, gain and omission of the unmodeled V3-VMed layer are H-level",
    ),
    "ZHANG2025_V3_MOTOR_GAIN": _source(
        "ZHANG2025_V3_MOTOR_GAIN", "10.1016/j.celrep.2024.115212",
        "broad V3 excitatory innervation and amplification of hindlimb motoneuron output",
        "mouse synaptophysin/VGLUT2 anatomy, in-vitro and awake optogenetic manipulation and genetic silencing",
        "population-level motor-output support does not specify V3 subpopulation, side, phase, connection weight, or convert the reported contact fraction/extensor functional bias into a model gain",
    ),
    "WANG2008_IA_RECIPROCAL": _source(
        "WANG2008_IA_RECIPROCAL", "10.1152/jn.90354.2008",
        "Ia reciprocal inhibition and Renshaw-to-IaIN circuit",
        "first-postnatal-week mouse Q-Ia-to-PBSt-MN in-vitro spinal preparation",
        "PF targets, reduced resource state and numeric gains are not measured",
    ),
    "GEERTSEN2011_IA_LOCOMOTION": _source(
        "GEERTSEN2011_IA_LOCOMOTION", "10.1113/jphysiol.2010.199125",
        "Ia inhibitory-interneuron activity in locomotion", "cat locomotor preparation",
        "species/phase transfer and exact model connections remain reduced",
    ),
    "WORTHY2024_V1_CLADES": _source(
        "WORTHY2024_V1_CLADES", "10.7554/eLife.95172",
        "V1 molecular clades relevant to IaIN identity", "mouse spinal molecular anatomy",
        "does not make the modeled V1Ia pool a pure genetic subtype",
    ),
    "LAMOTTE_DINCAMPS2017_MN_RC_MIXED": _source(
        "LAMOTTE_DINCAMPS2017_MN_RC_MIXED", "10.1038/s41598-017-04266-8",
        "mixed cholinergic/glutamatergic MN-to-Renshaw transmission",
        "P5-P10 mouse paired slices with P21-P90 ventral-root-evoked validation",
        "pre/post segregation is unresolved; child resources/delays and per-event dual processing are H-level",
    ),
    "PERRY2015_RENSHAW_IH_SK": _source(
        "PERRY2015_RENSHAW_IH_SK", "10.1111/ejn.12852",
        "Renshaw Ih/SK physiology", "mouse Renshaw cells",
        "exact reduced gates and conductances are not point estimates",
    ),
    "MOORE2015_RENSHAW_RECURRENT": _source(
        "MOORE2015_RENSHAW_RECURRENT", "10.1523/JNEUROSCI.2541-15.2015",
        "motoneuron-Renshaw recurrent circuit", "mouse spinal circuit",
        "exact probabilities, delays and weights remain priors",
    ),
    "ZHANG2014_V1_V2B_ALTERNATION": _source(
        "ZHANG2014_V1_V2B_ALTERNATION", "10.1016/j.neuron.2014.02.013",
        "V1/V2b flexor-extensor alternation", "mouse spinal genetics",
        "does not specify all RG/PF targets or numeric bias",
    ),
    "BRITZ2015_V1_V2B_ASYMMETRY": _source(
        "BRITZ2015_V1_V2B_ASYMMETRY", "10.7554/eLife.04718",
        "V1/V2b motor-pool targeting asymmetry", "mouse spinal anatomy and physiology",
        "exact 0.5 flexor gain and upstream targets are H-level",
    ),
    "DESTEXHE1994_SYN_KINETIC": _source(
        "DESTEXHE1994_SYN_KINETIC", "10.1162/neco.1994.6.1.14",
        "computational receptor-binding conductance reduction", "computational method",
        "not experimental spinal receptor identity, delay or parameter evidence",
    ),
    "BARRETT_STEVENS1972_NMJ_RELEASE": _source(
        "BARRETT_STEVENS1972_NMJ_RELEASE", "10.1113/jphysiol.1972.sp010054",
        "quantal NMJ release kinetics", "frog neuromuscular junction",
        "does not establish a fixed 0.50-ms model delay",
    ),
    "BUKHARAEVA2007_NMJ_RELEASE_SYNC": _source(
        "BUKHARAEVA2007_NMJ_RELEASE_SYNC", "10.1111/j.1471-4159.2006.04282.x",
        "NMJ release synchrony", "mouse neuromuscular junction",
        "does not identify the model's one-pool recovery law",
    ),
    "RICHARDS2003_NMJ_VESICLE_POOLS": _source(
        "RICHARDS2003_NMJ_VESICLE_POOLS", "10.1016/S0896-6273(03)00405-7",
        "NMJ readily releasable-pool recycling/refill", "frog neuromuscular junction",
        "does not support reserve-pool mobilization; normalized resource and refill constants remain priors",
    ),
    "REDMAN2022_NMJ_ACHE": _source(
        "REDMAN2022_NMJ_ACHE", "10.1111/bph.15940",
        "AChE-dependent NMJ signal clearance", "mouse neuromuscular junction",
        "single-exponential ACh clearance is a reduction",
    ),
    "ANDERSON_STEVENS1973_NMJ_CHANNEL": _source(
        "ANDERSON_STEVENS1973_NMJ_CHANNEL", "10.1113/jphysiol.1973.sp010410",
        "ACh-sensitive endplate channel conductance/closing", "frog neuromuscular junction",
        "Hill binding, sigmoid fiber threshold and gains are H-level",
    ),
    "ASHLEY_RIDGWAY1968_MUSCLE_CA_TENSION": _source(
        "ASHLEY_RIDGWAY1968_MUSCLE_CA_TENSION", "10.1038/2191168a0",
        "action-potential, Ca transient and tension sequence", "barnacle single muscle fibre",
        "qualitative cross-species mechanism only",
    ),
    "RIOS_BRUM1987_EC_COUPLING": _source(
        "RIOS_BRUM1987_EC_COUPLING", "10.1038/325717a0",
        "DHPR/SR excitation-contraction coupling", "skeletal-muscle preparation",
        "does not fix normalized model kinetics",
    ),
    "KONISHI_WATANABE1998_CA_FORCE": _source(
        "KONISHI_WATANABE1998_CA_FORCE", "10.1085/jgp.111.4.505",
        "steady Ca-force saturation", "intact frog skeletal muscle",
        "linear activation-force scale and extensor prior are not measured",
    ),
    "HUNTER_KEARNEY1982_ANKLE_DYNAMICS": _source(
        "HUNTER_KEARNEY1982_ANKLE_DYNAMICS", "10.1016/0021-9290(82)90089-6",
        "second-order inertia/viscous/elastic joint form", "human ankle",
        "one-DOF transfer and external-resistance multiplier are H-level",
    ),
    "MILEUSNIC2006_SPINDLE_MODEL": _source(
        "MILEUSNIC2006_SPINDLE_MODEL", "10.1152/jn.00868.2005",
        "original computational muscle-spindle transducer", "cat afferent data fit",
        "clipped linear proxy and model gains are H-level",
    ),
    "MILEUSNIC_LOEB2006_GTO_MODEL": _source(
        "MILEUSNIC_LOEB2006_GTO_MODEL", "10.1152/jn.00869.2005",
        "original computational GTO transducer", "cat afferent data fit",
        "clipped force proxy and central reflex sign are not established",
    ),
    "JANKOWSKA_MCCREA1983_IB_PATHS": _source(
        "JANKOWSKA_MCCREA1983_IB_PATHS", "10.1113/jphysiol.1983.sp014663",
        "polysynaptic group-I/Ib premotor pathways with EPSPs and IPSPs", "cat spinal cord",
        "does not establish direct Ib-to-PF/MN excitation",
    ),
    "PEARSON_COLLINS1993_IB_REVERSAL": _source(
        "PEARSON_COLLINS1993_IB_REVERSAL", "10.1152/jn.1993.70.3.1009",
        "locomotor-state-dependent Ib reflex reversal", "cat locomotion",
        "fixed positive sign is only a represented-regime hypothesis",
    ),
    "GOSSARD1994_IB_LOCOMOTOR": _source(
        "GOSSARD1994_IB_LOCOMOTOR", "10.1007/BF00228410",
        "phase-modulated Ib-dominant spinal pathway", "cat locomotor preparation",
        "identity of collapsed interneurons and exact gains remain unresolved",
    ),
    "QU2019_PRESYN_MT": _source(
        "QU2019_PRESYN_MT", "10.1016/j.cub.2019.10.049",
        "activity-induced presynaptic MT nucleation/SV transport", "rat culture and mouse hippocampal vGlut1 boutons",
        "not spinal or inhibitory terminals; universal route transfer is H-level",
    ),
    "BABU2020_MT_RECOVERY": _source(
        "BABU2020_MT_RECOVERY", "10.1523/JNEUROSCI.1571-19.2019",
        "MT-dependent slow synaptic recovery", "rat calyx of Held",
        "combining it with Qu into one all-route spinal chain is H-level",
    ),
}


def _binding(
    mechanism_grade: str,
    source_ids: Sequence[str],
    supported_scope: str,
    h_level_boundary: str,
    realization_grade: str = "H",
) -> EvidenceBinding:
    return EvidenceBinding(
        mechanism_grade, realization_grade, tuple(source_ids),
        supported_scope, h_level_boundary,
    )


CELL_CLASS_EVIDENCE: Dict[str, EvidenceBinding] = {
    "RG": _binding("B", ("SINGH2025_SHOX2_CURRENTS", "HA_DOUGHERTY2018_SHOX2_COUPLING", "DOUGHERTY2013_SHOX2_RHYTHM"), "Shox2-enriched rhythmogenic ensemble with measured current subsets and neonatal coupling", "RG membership, all numeric parameters and exact target scaffold remain H-level"),
    "PF": _binding("H", ("LAFRENIERE_ROULA2005_CPG_DELETIONS", "ZHONG2012_MOUSE_CPG_DELETIONS"), "experimental rhythm-pattern separation motivates the architecture but does not identify the modeled PF entity", "PF class identity, slow state, topology and parameters are H-level"),
    "MN": _binding("A", ("LAMOTTE_DINCAMPS2017_MN_RC_MIXED", "CHOPEK2018_V3_MICROCIRCUIT", "CARLIN2000_MN_DEND_LCA", "LI_BENNETT2003_MN_NAP_LCA_PIC", "SCHWINDT_CRILL1980_MN_PIC", "LI_BENNETT2007_MN_SK"), "alpha-MN PIC/SK mechanisms, mixed Renshaw output and glutamatergic recurrent excitation of ventral V3", "two-compartment placement, gates, gains, child resources/delays and exact V3-VLat target allocation are H-level"),
    "V0D": _binding("B", ("LANUZA2004_V0_COORDINATION", "TALPALAR2013_V0_SPEED"), "inhibitory commissural V0D role", "exact RG target, phase and gains are H-level"),
    "V0V": _binding("B", ("MORAN_RIVARD2001_V0V_IDENTITY", "CRONE2008_V2A_V0V", "TALPALAR2013_V0_SPEED"), "excitatory commissural V0V identity/high-speed motif", "exact V1Ia target, phase and gains are H-level"),
    "V2a": _binding("A", ("ZHONG2010_V2A_PHENOTYPES", "CRONE2008_V2A_V0V", "HAYASHI2018_V2A_ARRAYS"), "measured firing/sag phenotypes, same-phenotype coupling, V0V motif and population-level ipsilateral MN output", "AdEx adaptation/delay realization, masks, gates, PF target and tonic/phasic/delayed contribution to the MN path are H-level; the Hayashi molecular/connectivity Type-I/Type-II axis is not equated to the Zhong firing phenotypes"),
    "V3": _binding("B", ("BOROWSKA2013_V3_SUBPOPS", "ZHANG2008_V3_BALANCE", "CHOPEK2018_V3_MICROCIRCUIT", "ZHANG2025_V3_MOTOR_GAIN"), "Sim1-positive V3 identity/dorsoventral intrinsic physiology, direct V3-VLat ipsilateral motor microcircuit and broad/predominantly contralateral motor-pool targeting", "the class-level grade is the lowest supported component; intrinsic mechanisms and motor pathways retain separate A/B claim grades, while the connectivity-only V3-VLat fraction, subtype/phase allocation, gates and exact parameters are H-level"),
    "V1Ia": _binding("B", ("WANG2008_IA_RECIPROCAL", "GEERTSEN2011_IA_LOCOMOTION", "WORTHY2024_V1_CLADES"), "V1-enriched mapping of the physiological reciprocal-Ia-interneuron pool", "pool purity and class mapping are transferred; direct circuit components retain separate A-grade term/pathway claims"),
    "V1Ren": _binding("A", ("LAMOTTE_DINCAMPS2017_MN_RC_MIXED", "WANG2008_IA_RECIPROCAL", "PERRY2015_RENSHAW_IH_SK", "MOORE2015_RENSHAW_RECURRENT"), "Renshaw recurrent circuit, mixed MN input, Ih/SK and IaIN output", "reduced receptor/gate kinetics and child release states are H-level"),
    "V2b": _binding("B", ("ZHANG2014_V1_V2B_ALTERNATION", "BRITZ2015_V1_V2B_ASYMMETRY"), "V2b alternation and motor-target bias", "RG/PF targets and exact 0.5 flexor gain are H-level"),
}


INTRINSIC_TERM_EVIDENCE: Dict[str, EvidenceBinding] = {
    **{
        term: _binding("C", ("BRETTE_GERSTNER2005_ADEX",), "generic AdEx membrane term", "all class parameters are reduced model priors")
        for term in COMMON_ADEX_RUNTIME_TERM_IDS
    },
    **{
        term: _binding("B", ("SINGH2025_SHOX2_CURRENTS",), "current family/subset measured in heterogeneous Shox2 samples and transferred to the functional RG ensemble", "neonatal masks, shared PIC mask, gates, co-occurrence and gains are H-level reductions")
        for term in ("I_NAP_RG", "I_LTYPE_CA_PIC_RG", "I_M_RG", "I_KCA_RG", "I_H_RG", "I_T_RG", "I_A_RG")
    },
    "I_A_RG": _binding("B", ("SINGH2025_SHOX2_CURRENTS",), "neonatal Shox2 IA prevalence is approximately 14%; 5/35 is inferred from Figure 1K and the reported total rather than stated in prose", "transfer to functional RG, exact numerator, mask, gates and gain are H-level"),
    "I_RG_GAP": _binding("A", ("HA_DOUGHERTY2018_SHOX2_COUPLING",), "P0-5 within-Shox2 non-V2a electrical-pair incidence 6/18", "nearby-pair sampling transfer, sparse graph realization and conductance are H-level"),
    "I_PF_SLOW": _binding("H", ("LAFRENIERE_ROULA2005_CPG_DELETIONS", "ZHONG2012_MOUSE_CPG_DELETIONS"), "functional slow pattern-formation integration", "no PF-specific molecular current is claimed"),
    "I_MN_DEND_LEAK": _binding("H", ("CARLIN2000_MN_DEND_LCA",), "reduced dendritic compartment", "compartment geometry and leak are H-level"),
    "I_MN_NAP_PIC": _binding("A", ("LI_BENNETT2003_MN_NAP_LCA_PIC", "SCHWINDT_CRILL1980_MN_PIC"), "motoneuron Na-like PIC", "gate and conductance are H-level"),
    "I_MN_LTYPE_CA_PIC": _binding("A", ("CARLIN2000_MN_DEND_LCA", "LI_BENNETT2003_MN_NAP_LCA_PIC"), "motoneuron dendritic L-type-Ca PIC", "gate, reversal and conductance are H-level"),
    "I_MN_COUPLING_SOMA": _binding("H", ("CARLIN2000_MN_DEND_LCA",), "reduced soma-dendrite exchange", "two-compartment coupling coefficient is H-level"),
    "I_MN_AHP": _binding("A", ("LI_BENNETT2007_MN_SK",), "motoneuron SK/mAHP", "normalized Ca gate and conductance are H-level"),
    **{
        term: _binding("B", ("ZHONG2010_V2A_PHENOTYPES",), "V2a sag/rebound is Ih-consistent current-clamp evidence", "HCN identity, deterministic masks, gates and gains are H-level")
        for term in ("I_V2A_H_TONIC", "I_V2A_H_PHASIC", "I_V2A_H_DELAYED")
    },
    "I_V2A_DELAY": _binding("H", ("ZHONG2010_V2A_PHENOTYPES",), "delayed-onset firing phenotype is measured", "the effective outward-current identity and kinetics are H-level"),
    "I_V2A_GAP_TONIC": _binding("A", ("ZHONG2010_V2A_PHENOTYPES",), "tonic-tonic electrical pair incidence 13/47", "sparse pair sampling and conductance are H-level"),
    "I_V2A_GAP_PHASIC": _binding("A", ("ZHONG2010_V2A_PHENOTYPES",), "phasic-phasic electrical pair incidence 3/7", "sparse pair sampling and conductance are H-level"),
    **{
        term: _binding("A", ("BOROWSKA2013_V3_SUBPOPS",), "V3 dorsoventral Ih/T-type physiology", "subtype masks, gates and gains are H-level")
        for term in ("I_V3_H_VENTRAL", "I_V3_H_DORSAL", "I_V3_T_DORSAL")
    },
    "I_RENSHAW_H": _binding("A", ("PERRY2015_RENSHAW_IH_SK",), "Renshaw Ih", "gate and conductance are H-level"),
    "I_RENSHAW_SK": _binding("A", ("PERRY2015_RENSHAW_IH_SK",), "Renshaw SK/AHP", "normalized Ca state and conductance are H-level"),
}


DIRECT_INPUT_EVIDENCE: Dict[str, EvidenceBinding] = {
    "I_TONIC_CLASS": _binding("H", (), "class baseline excitability prior", "not a measured current or hidden outcome target"),
    "I_DESCENDING_RG": _binding("H", ("DOUGHERTY2013_SHOX2_RHYTHM",), "single descending command restricted to RG", "amplitudes are H-level and contain no target frequency"),
    "I_PERTURBATION": _binding("H", (), "predeclared exogenous perturbation", "protocol manipulation, not endogenous biology"),
    "I_IA_TO_PF_EFFECTIVE": _binding("H", ("MILEUSNIC2006_SPINDLE_MODEL", "WANG2008_IA_RECIPROCAL"), "Ia-like sensory context applied to the latent PF module", "PF target, clipped proxy, resource and gain are H-level"),
    "I_IA_TO_MN": _binding("A", ("WANG2008_IA_RECIPROCAL",), "homonymous Ia-to-MN excitatory motif", "the separately sourced clipped transducer proxy, reduced resource and gain are H-level"),
    "I_IA_TO_V1IA": _binding("A", ("WANG2008_IA_RECIPROCAL",), "Ia recruitment of the reciprocal Ia inhibitory circuit", "pool reduction, the separately sourced clipped transducer proxy, resource and gain are H-level"),
    "I_IB_TO_PF_EFFECTIVE": _binding("H", ("MILEUSNIC_LOEB2006_GTO_MODEL", "JANKOWSKA_MCCREA1983_IB_PATHS", "PEARSON_COLLINS1993_IB_REVERSAL", "GOSSARD1994_IB_LOCOMOTOR"), "context-dependent group-I/Ib physiology collapsed onto the latent PF module", "fixed positive represented-regime sign, PF target and gain are H-level"),
    "I_IB_TO_MN_EFFECTIVE": _binding("H", ("MILEUSNIC_LOEB2006_GTO_MODEL", "JANKOWSKA_MCCREA1983_IB_PATHS", "PEARSON_COLLINS1993_IB_REVERSAL", "GOSSARD1994_IB_LOCOMOTOR"), "context-dependent polysynaptic group-I/Ib physiology collapsed onto MN current", "not monosynaptic Ib excitation; fixed sign, target reduction and gain are H-level"),
    "I_PF_DELETION": _binding("H", ("LAFRENIERE_ROULA2005_CPG_DELETIONS", "ZHONG2012_MOUSE_CPG_DELETIONS"), "predeclared PF deletion intervention", "current magnitude is a protocol prior"),
}


PERIPHERAL_OUTPUT_EVIDENCE: Dict[str, EvidenceBinding] = {
    "E_NMJ_VESICLE_ACH": _binding("A", ("BARRETT_STEVENS1972_NMJ_RELEASE", "BUKHARAEVA2007_NMJ_RELEASE_SYNC", "RICHARDS2003_NMJ_VESICLE_POOLS", "REDMAN2022_NMJ_ACHE"), "quantal release, vesicle-pool/refill observations and ACh clearance", "population-lumped junction, fixed delay, one-pool law, exponential clearance and gains are H-level; postsynaptic endplate conductance is a separate claim"),
}


INTERFACE_EVIDENCE: Dict[str, EvidenceBinding] = {
    "central_chemical_synapse": _binding("C", ("DESTEXHE1994_SYN_KINETIC",), "kinetic conductance reduction", "spinal receptor identity, one-state decay, delays and gains are H-level"),
    "renshaw_mixed_mn_input": _binding("A", ("LAMOTTE_DINCAMPS2017_MN_RC_MIXED",), "same MN-Renshaw pair has ACh and glutamate components", "pre/post segregation, child resources/delays and per-event dual processing are H-level"),
    "nmj_presynaptic_release": _binding("A", ("BARRETT_STEVENS1972_NMJ_RELEASE", "BUKHARAEVA2007_NMJ_RELEASE_SYNC", "RICHARDS2003_NMJ_VESICLE_POOLS", "REDMAN2022_NMJ_ACHE"), "NMJ release/pool/clearance mechanisms", "fixed 0.50 ms, one normalized pool and exponential ACh are H-level"),
    "nmj_postsynaptic_endplate": _binding("A", ("ANDERSON_STEVENS1973_NMJ_CHANNEL",), "ACh-sensitive endplate conductance", "Hill binding, sigmoid threshold and gains are H-level"),
    "excitation_contraction": _binding("A", ("ASHLEY_RIDGWAY1968_MUSCLE_CA_TENSION", "RIOS_BRUM1987_EC_COUPLING", "KONISHI_WATANABE1998_CA_FORCE"), "AP-to-SR-Ca-to-activation/force sequence", "cross-species normalized states and kinetics are H-level"),
    "muscle_force": _binding("A", ("KONISHI_WATANABE1998_CA_FORCE",), "Ca-force saturation", "linear normalized active-force map and extensor scale prior are H-level"),
    "joint_mechanics": _binding("B", ("HUNTER_KEARNEY1982_ANKLE_DYNAMICS",), "second-order inertia/damping/stiffness form", "one-DOF transfer and load multiplier are H-level"),
    "Ia_spindle_transducer": _binding("C", ("MILEUSNIC2006_SPINDLE_MODEL",), "spindle transduction provenance", "bounded length/velocity proxy is not a complete spindle model"),
    "Ia_effective_spinal_pathway": _binding("H", ("WANG2008_IA_RECIPROCAL",), "direct Ia-to-MN/IaIN observations motivate the interface", "the combined executable interface also contains an unsupported PF target plus reduced resource and gains"),
    "Ib_GTO_transducer": _binding("C", ("MILEUSNIC_LOEB2006_GTO_MODEL",), "GTO transduction provenance", "bounded linear force proxy is H-level"),
    "Ib_effective_spinal_pathway": _binding("H", ("JANKOWSKA_MCCREA1983_IB_PATHS", "PEARSON_COLLINS1993_IB_REVERSAL", "GOSSARD1994_IB_LOCOMOTOR"), "context-dependent polysynaptic locomotor group-I/Ib observations motivate the interface", "the executable collapsed positive PF/MN action and extensor gain are H-level"),
    "local_mt_terminal": _binding("H", ("QU2019_PRESYN_MT", "BABU2020_MT_RECOVERY"), "presynaptic activity-MT/SV and slow-recovery observations motivate the interface", "the executable chain combines two nonspinal glutamatergic domains, extends them only across every modeled central chemical edge/source-class route, and adds a depleting normalized slow-replenishment resource; NMJ and sensory paths are separate and no anatomical reserve-pool mobilization is claimed"),
}


# The prose-level evidence label on each human-readable interface record is
# frozen independently from the structured A/B/C/H binding. This deliberate
# duplication makes a one-sided metadata edit fail closed instead of silently
# changing the biological claim.
INTERFACE_RECORD_EVIDENCE_CONTRACT: Dict[str, Tuple[str, str]] = {
    "central_chemical_synapse": (
        "computational_kinetic_provenance_not_spinal_receptor_identity", "C",
    ),
    "renshaw_mixed_mn_input": (
        "direct_single_pair_mixed_input_and_transmitter_system_segregation_supported_component_edge_resource_delay_receptor_kinetics_and_equal_gain_are_H_level", "A",
    ),
    "nmj_presynaptic_release": (
        "mechanisms_supported_single_pool_rates_and_gain_are_H_level", "A",
    ),
    "nmj_postsynaptic_endplate": (
        "endplate_channel_supported_Hill_binding_sigmoid_fiber_threshold_and_gains_are_H_level", "A",
    ),
    "excitation_contraction": (
        "mechanism_supported_gain_is_model_prior", "A",
    ),
    "muscle_force": (
        "phenomenological_normalized_force_gain_extensor_scale_is_H_level", "A",
    ),
    "joint_mechanics": (
        "inertia_damping_stiffness_supported_external_resistance_multiplier_is_H_level", "B",
    ),
    "Ia_spindle_transducer": (
        "primary_sensor_reduction_not_a_complete_spindle_model", "C",
    ),
    "Ia_effective_spinal_pathway": (
        "mixed_direct_circuit_support_PF_and_gains_are_H_level", "H",
    ),
    "Ib_GTO_transducer": (
        "primary_sensor_reduction_not_a_central_reflex_sign", "C",
    ),
    "Ib_effective_spinal_pathway": (
        "H_level_effective_path_not_monosynaptic_Ib_excitation", "H",
    ),
    "local_mt_terminal": (
        "two_nonspinal_domains_combined_all_modeled_central_chemical_routes_and_normalized_resource_law_are_H_level", "H",
    ),
}


PATHWAY_EVIDENCE: Dict[str, EvidenceBinding] = {
    "RG_recurrent": _binding("H", ("HA_DOUGHERTY2018_SHOX2_COUPLING",), "broad within-Shox2 connectivity", "exact chemical recurrent RG graph is H-level"),
    "RG_to_V1Ia": _binding("H", ("WANG2008_IA_RECIPROCAL", "DOUGHERTY2013_SHOX2_RHYTHM"), "RG/IaIN functional context", "exact direct target is H-level"),
    "V1Ia_to_antagonist_RG": _binding("H", ("WANG2008_IA_RECIPROCAL", "GEERTSEN2011_IA_LOCOMOTION"), "reciprocal inhibitory motif", "RG target is H-level"),
    "RG_to_V2b": _binding("H", ("ZHANG2014_V1_V2B_ALTERNATION",), "V2b locomotor recruitment context", "exact direct RG target is H-level"),
    "V2b_to_antagonist_RG": _binding("H", ("ZHANG2014_V1_V2B_ALTERNATION",), "V2b flexor-extensor motif", "RG target/gain is H-level"),
    "RG_to_PF": _binding("H", ("LAFRENIERE_ROULA2005_CPG_DELETIONS", "ZHONG2012_MOUSE_CPG_DELETIONS"), "rhythm-to-pattern functional architecture", "PF module edge is H-level"),
    "PF_recurrent": _binding("H", ("LAFRENIERE_ROULA2005_CPG_DELETIONS", "ZHONG2012_MOUSE_CPG_DELETIONS"), "pattern-layer persistence motif", "exact recurrent PF graph is H-level"),
    "V1Ia_to_antagonist_PF": _binding("H", ("WANG2008_IA_RECIPROCAL",), "reciprocal inhibitory motif", "PF target is H-level"),
    "V2b_to_antagonist_PF": _binding("H", ("ZHANG2014_V1_V2B_ALTERNATION",), "V2b inhibitory motif", "PF target is H-level"),
    "PF_to_MN": _binding("H", ("LAFRENIERE_ROULA2005_CPG_DELETIONS", "ZHONG2012_MOUSE_CPG_DELETIONS"), "pattern-to-motor functional transfer", "latent PF-to-MN edge is H-level"),
    "V1Ia_to_antagonist_MN": _binding("A", ("WANG2008_IA_RECIPROCAL", "GEERTSEN2011_IA_LOCOMOTION"), "reciprocal Ia inhibition of antagonist MN", "probability, delay and gain are H-level"),
    "V2b_to_antagonist_MN": _binding("B", ("ZHANG2014_V1_V2B_ALTERNATION", "BRITZ2015_V1_V2B_ASYMMETRY"), "V2b motor inhibition and extensor bias", "exact 0.5 flexor gain/probability/delay are H-level"),
    "MN_to_V1Ren_nAChR": _binding("A", ("LAMOTTE_DINCAMPS2017_MN_RC_MIXED",), "cholinergic component in same MN-Renshaw pair", "child edge/resource/delay and per-event processing are H-level"),
    "MN_to_V1Ren_GluR": _binding("A", ("LAMOTTE_DINCAMPS2017_MN_RC_MIXED",), "glutamatergic component in same MN-Renshaw pair", "child edge/resource/delay and per-event processing are H-level"),
    "V1Ren_to_MN": _binding("A", ("MOORE2015_RENSHAW_RECURRENT",), "recurrent Renshaw inhibition of MN", "probability, delay and gain are H-level"),
    "V1Ren_to_V1Ia": _binding("A", ("WANG2008_IA_RECIPROCAL",), "Renshaw inhibition of IaIN", "probability, delay and gain are H-level"),
    "RG_to_V2a": _binding("H", ("DOUGHERTY2013_SHOX2_RHYTHM", "ZHONG2010_V2A_PHENOTYPES"), "Shox2/V2a locomotor context", "exact RG-to-V2a edge is H-level"),
    "V2a_to_V0V": _binding("A", ("CRONE2008_V2A_V0V",), "direct excitatory V2a contacts onto molecular V0 commissural neurons", "probability, delay and gain are H-level"),
    "V2a_to_PF": _binding("H", ("CRONE2008_V2A_V0V",), "V2a ipsilateral excitatory role", "PF target is H-level"),
    "V2a_to_MN": _binding("A", ("HAYASHI2018_V2A_ARRAYS",), "population-level ipsilateral V2a-to-motoneuron excitatory motif", "mapping tonic/phasic/delayed phenotypes onto Hayashi Type-I/II, exact subtype contributions, probability, delay and gain are H-level"),
    "V0V_to_cross_V1Ia": _binding("H", ("LANUZA2004_V0_COORDINATION", "TALPALAR2013_V0_SPEED"), "crossed-disinhibitory V0V motif", "exact V1Ia target and phase are H-level"),
    "RG_to_V0D": _binding("H", ("LANUZA2004_V0_COORDINATION",), "V0D commissural recruitment context", "exact RG-to-V0D edge is H-level"),
    "V0D_cross_inhibition": _binding("H", ("LANUZA2004_V0_COORDINATION", "TALPALAR2013_V0_SPEED"), "inhibitory commissural V0D motif", "exact contralateral RG target/phase is H-level"),
    "RG_to_V3": _binding("H", ("ZHANG2008_V3_BALANCE",), "V3 bilateral-balance recruitment context", "exact RG-to-V3 edge is H-level"),
    "V3_ventral_to_contralateral_MN_flexor": _binding("B", ("ZHANG2008_V3_BALANCE", "BOROWSKA2013_V3_SUBPOPS"), "ventral V3 contribution to predominantly contralateral iliopsoas flexor motor-pool input", "ventral source restriction combines subtype/topology evidence; probability, delay, current and online phase assignment are H-level"),
    "V3_ventral_to_contralateral_MN_extensor": _binding("B", ("ZHANG2008_V3_BALANCE", "BOROWSKA2013_V3_SUBPOPS"), "ventral V3 contribution to predominantly contralateral gastrocnemius extensor motor-pool input", "ventral source restriction combines subtype/topology evidence; probability, delay, current and online phase assignment are H-level"),
    "V3_VLat_to_ipsilateral_MN": _binding("A", ("CHOPEK2018_V3_MICROCIRCUIT",), "ventrolateral V3 excitation of ipsilateral motoneurons", "the connectivity-only V3-VLat mask fraction, flexor/extensor phase assignment, probability, delay and gain are H-level"),
    "MN_to_V3_VLat_GluR": _binding("A", ("CHOPEK2018_V3_MICROCIRCUIT",), "recurrent glutamatergic motoneuron excitation of ventral/ventrolateral V3", "the exact V3-VLat target restriction, receptor reduction, phase assignment, probability, delay and gain are H-level"),
}


def literature_evidence_contract_payload() -> Dict[str, object]:
    """Return the canonical JSON-ready claim-to-source contract.

    Its digest is stored in every simulation manifest.  Therefore a source,
    evidence grade, preparation boundary, executable-term binding, or H-level
    realization boundary cannot drift without changing simulation identity.
    """
    return {
        "schema": "source-resolved-biological-claims-1.0",
        "grade_semantics": {
            "A": "direct_primary_experimental_mechanism_or_circuit_evidence",
            "B": "identity_function_or_motif_evidence_with_transfer",
            "C": "original_computational_or_functional_architecture_provenance",
            "H": "explicit_model_hypothesis_or_exact_numerical_realization",
        },
        "sources": {
            name: asdict(record)
            for name, record in sorted(LITERATURE_SOURCES.items())
        },
        "claim_registries": {
            registry_name: {
                claim_id: asdict(binding)
                for claim_id, binding in sorted(registry.items())
            }
            for registry_name, registry in (
                ("cell_class", CELL_CLASS_EVIDENCE),
                ("intrinsic_term", INTRINSIC_TERM_EVIDENCE),
                ("direct_input", DIRECT_INPUT_EVIDENCE),
                ("peripheral_output", PERIPHERAL_OUTPUT_EVIDENCE),
                ("biological_interface", INTERFACE_EVIDENCE),
                ("pathway", PATHWAY_EVIDENCE),
            )
        },
        "interface_record_evidence_contract": {
            interface: {
                "evidence_level": evidence_level,
                "mechanism_evidence": mechanism_evidence,
            }
            for interface, (evidence_level, mechanism_evidence) in sorted(
                INTERFACE_RECORD_EVIDENCE_CONTRACT.items()
            )
        },
        "connectivity_subphenotype_contracts": {
            name: asdict(contract)
            for name, contract in sorted(
                V3_CONNECTIVITY_SUBPHENOTYPE_CONTRACTS.items()
            )
        },
    }


def validate_literature_evidence_contract() -> None:
    """Require one source-resolved A/B/C/H contract for every named claim."""
    if set(LITERATURE_SOURCES) != {
        record.source_id for record in LITERATURE_SOURCES.values()
    }:
        raise ValueError("Literature source key/source_id mismatch")
    if set(LITERATURE_SOURCE_TITLES) != set(LITERATURE_SOURCES):
        raise ValueError("Literature title/source registry mismatch")
    source_urls = [record.url for record in LITERATURE_SOURCES.values()]
    if len(source_urls) != len(set(source_urls)):
        raise ValueError("Literature sources must have unique canonical URLs")
    for source_id, source in LITERATURE_SOURCES.items():
        if not source.url.startswith("https://doi.org/"):
            raise ValueError(f"{source_id} lacks a canonical DOI URL")
        if source.title != LITERATURE_SOURCE_TITLES[source_id]:
            raise ValueError(f"{source_id} canonical title drift")
        if not all((
            source.title, source.evidence_domain, source.preparation,
            source.scope_limit,
        )):
            raise ValueError(f"{source_id} has incomplete source scope")
    for name, contract in V3_CONNECTIVITY_SUBPHENOTYPE_CONTRACTS.items():
        if name != contract.subphenotype:
            raise ValueError("Connectivity-subphenotype key/name drift")
        if contract.evidence_grade not in {"A", "B", "C", "H"}:
            raise ValueError(f"{name} has an invalid subphenotype evidence grade")
        if not contract.source_ids or set(contract.source_ids) - set(
            LITERATURE_SOURCES
        ):
            raise ValueError(f"{name} has invalid subphenotype sources")

    declared_pathways = set()
    for contract in CLASS_EXECUTION_CONTRACTS.values():
        declared_pathways.update(contract.incoming_pathways.all_names())
        declared_pathways.update(contract.outgoing_pathways.all_names())
    exact_registries = (
        ("cell class", CELL_CLASS_EVIDENCE, set(CLASSES)),
        ("intrinsic term", INTRINSIC_TERM_EVIDENCE, RUNTIME_INTRINSIC_TERM_IDS),
        ("direct input", DIRECT_INPUT_EVIDENCE, DIRECT_INPUT_TERM_IDS),
        ("peripheral output", PERIPHERAL_OUTPUT_EVIDENCE, PERIPHERAL_OUTPUT_TERM_IDS),
        ("biological interface", INTERFACE_EVIDENCE, set(BIOLOGICAL_INTERFACE_EQUATIONS)),
        ("pathway", PATHWAY_EVIDENCE, declared_pathways),
    )
    all_bindings: List[Tuple[str, str, EvidenceBinding]] = []
    for kind, registry, expected in exact_registries:
        if set(registry) != set(expected):
            raise ValueError(
                f"{kind} evidence registry mismatch: "
                f"missing={sorted(set(expected) - set(registry))}, "
                f"extra={sorted(set(registry) - set(expected))}"
            )
        all_bindings.extend((kind, claim_id, binding) for claim_id, binding in registry.items())
    for kind, claim_id, binding in all_bindings:
        if binding.mechanism_evidence not in {"A", "B", "C", "H"}:
            raise ValueError(f"{kind} {claim_id} has invalid mechanism grade")
        if binding.realization_evidence not in {"A", "B", "C", "H"}:
            raise ValueError(f"{kind} {claim_id} has invalid realization grade")
        if binding.realization_evidence != "H":
            raise ValueError(
                f"{kind} {claim_id} improperly treats exact numerical realization as measured"
            )
        if (
            binding.mechanism_evidence != "H"
            and not binding.source_ids
        ):
            raise ValueError(f"{kind} {claim_id} has evidence but no source")
        if len(binding.source_ids) != len(set(binding.source_ids)):
            raise ValueError(f"{kind} {claim_id} repeats a source")
        unknown = set(binding.source_ids) - set(LITERATURE_SOURCES)
        if unknown:
            raise ValueError(f"{kind} {claim_id} has unknown sources: {sorted(unknown)}")
        if not binding.supported_scope or not binding.h_level_boundary:
            raise ValueError(f"{kind} {claim_id} lacks explicit evidence/H boundary")

    # Human-readable record URLs and machine bindings are forced to be the same
    # set, preventing a class-wide citation from silently covering an unbound
    # term or a stale URL from surviving a source-contract revision.
    for cell_class, record in CELL_CLASS_EQUATIONS.items():
        bound_urls = {
            LITERATURE_SOURCES[source_id].url
            for source_id in CELL_CLASS_EVIDENCE[cell_class].source_ids
        }
        if set(record.literature_urls) != bound_urls:
            raise ValueError(f"{cell_class} class URL/evidence-source drift")
    for interface, record in BIOLOGICAL_INTERFACE_EQUATIONS.items():
        bound_urls = {
            LITERATURE_SOURCES[source_id].url
            for source_id in INTERFACE_EVIDENCE[interface].source_ids
        }
        if set(record.literature_urls) != bound_urls:
            raise ValueError(f"{interface} interface URL/evidence-source drift")
        expected_text, expected_grade = INTERFACE_RECORD_EVIDENCE_CONTRACT[interface]
        if record.evidence_level != expected_text:
            raise ValueError(f"{interface} interface evidence-level text drift")
        if INTERFACE_EVIDENCE[interface].mechanism_evidence != expected_grade:
            raise ValueError(f"{interface} interface evidence-grade drift")


def validate_biological_interface_equations() -> None:
    """Reject named biological interfaces without dynamics and evidence."""
    records = tuple(BIOLOGICAL_INTERFACE_EQUATIONS.values())
    if set(INTERFACE_RECORD_EVIDENCE_CONTRACT) != set(BIOLOGICAL_INTERFACE_EQUATIONS):
        raise ValueError("Biological interface evidence contract mismatch")
    if len({record.equation_id for record in records}) != len(records):
        raise ValueError("Every biological interface must have a unique equation_id")
    for key, record in BIOLOGICAL_INTERFACE_EQUATIONS.items():
        if key != record.interface:
            raise ValueError(f"Biological interface key mismatch: {key}")
        if not record.state_variables or not record.equation_terms:
            raise ValueError(f"{key} lacks explicit state/equation terms")
        if not record.input_role or not record.output_role:
            raise ValueError(f"{key} lacks an input/output mapping")
        if not record.literature_urls or not all(url.startswith("https://") for url in record.literature_urls):
            raise ValueError(f"{key} lacks literature traceability")
        if not record.implementation_symbols:
            raise ValueError(f"{key} lacks implementation symbols")
        expected_text, expected_grade = INTERFACE_RECORD_EVIDENCE_CONTRACT[key]
        if record.evidence_level != expected_text:
            raise ValueError(f"{key} interface evidence-level text drift")
        if INTERFACE_EVIDENCE[key].mechanism_evidence != expected_grade:
            raise ValueError(f"{key} interface evidence-grade drift")


@dataclass(frozen=True)
class Pathway:
    name: str
    source_population: int
    target_population: int
    population_weight_pa: float
    connection_probability: float
    delay_bins_ms: Tuple[float, ...]
    functional_role: str = "none"
    mt_route: str = "none"
    recruitment_axis: str = "none"
    evidence_class: str = "model_hypothesis"
    evidence_note: str = ""
    topology_group: str = "none"
    source_subphenotype: str = "all"
    target_subphenotype: str = "all"


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable logistic activation evaluated by one compiled ufunc."""
    return expit(x)


def fractional_delay_bins(
    crossing_fraction: float | np.ndarray,
    delay_ms: float | np.ndarray,
    dt_ms: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map a within-step event and physical delay to two grid endpoints.

    The returned lower/upper integer offsets and weights conserve event mass
    and its first temporal moment. ``crossing_fraction`` is measured from the
    start of the current integration interval and must lie in ``[0, 1]``.
    """
    if not math.isfinite(float(dt_ms)) or dt_ms <= 0.0:
        raise ValueError("dt_ms must be positive and finite")
    fractions, delays = np.broadcast_arrays(
        np.asarray(crossing_fraction, dtype=float),
        np.asarray(delay_ms, dtype=float),
    )
    if not (
        np.all(np.isfinite(fractions))
        and np.all((0.0 <= fractions) & (fractions <= 1.0))
    ):
        raise ValueError("crossing_fraction must be finite and in [0, 1]")
    if not np.all(np.isfinite(delays) & (delays >= 0.0)):
        raise ValueError("delay_ms must be finite and nonnegative")
    offsets = fractions + delays / dt_ms
    nearest = np.rint(offsets)
    offsets = np.where(np.abs(offsets - nearest) <= 1.0e-12, nearest, offsets)
    lower = np.floor(offsets).astype(np.int64)
    upper = lower + 1
    upper_weight = offsets - lower
    lower_weight = 1.0 - upper_weight
    return lower, upper, lower_weight, upper_weight


def right_endpoint_event_decay(
    crossing_fraction: float | np.ndarray,
    dt_ms: float,
    tau_ms: float,
) -> np.ndarray:
    """Age an impulse from its within-step time to the common right endpoint."""
    fractions = np.asarray(crossing_fraction, dtype=float)
    if not math.isfinite(float(dt_ms)) or dt_ms <= 0.0:
        raise ValueError("dt_ms must be positive and finite")
    if not math.isfinite(float(tau_ms)) or tau_ms <= 0.0:
        raise ValueError("tau_ms must be positive and finite")
    if not (
        np.all(np.isfinite(fractions))
        and np.all((0.0 <= fractions) & (fractions <= 1.0))
    ):
        raise ValueError("crossing_fraction must be finite and in [0, 1]")
    return np.exp(-(1.0 - fractions) * dt_ms / tau_ms)


def exponential_rosenbrock_increment(
    rhs_per_ms: np.ndarray,
    jacobian_per_ms: np.ndarray,
    interval_ms: float | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate a locally linearized scalar ODE over one physical interval.

    For ``dx/dt=f(x)`` with frozen local Jacobian ``J``, the increment is
    ``dt*phi_1(dt*J)*f``.  The returned second array is the effective Jacobian
    used after a conservative exponential-argument cap; it allows threshold
    time inversion to remain exactly consistent with the voltage increment.
    This is a numerical integration rule only and introduces no biological
    state, current or feedback.
    """
    rhs, jacobian, interval = np.broadcast_arrays(
        np.asarray(rhs_per_ms, dtype=float),
        np.asarray(jacobian_per_ms, dtype=float),
        np.asarray(interval_ms, dtype=float),
    )
    if not (
        np.all(np.isfinite(rhs))
        and np.all(np.isfinite(jacobian))
        and np.all(np.isfinite(interval))
        and np.all(interval >= 0.0)
    ):
        raise ValueError("Rosenbrock inputs must be finite and interval nonnegative")
    raw_argument = jacobian * interval
    argument = np.clip(raw_argument, -50.0, 50.0)
    effective_jacobian = np.divide(
        argument,
        interval,
        out=jacobian.copy(),
        where=interval > 0.0,
    )
    phi1 = np.empty_like(argument)
    small = np.abs(argument) <= 1.0e-7
    phi1[small] = (
        1.0 + argument[small] / 2.0 + argument[small] ** 2 / 6.0
    )
    phi1[~small] = np.expm1(argument[~small]) / argument[~small]
    return interval * rhs * phi1, effective_jacobian


def locally_linearized_threshold_fraction(
    initial_voltage_mv: np.ndarray,
    threshold_mv: float,
    rhs_per_ms: np.ndarray,
    effective_jacobian_per_ms: np.ndarray,
    active_interval_ms: np.ndarray,
    active_start_fraction: np.ndarray,
    outer_dt_ms: float,
    endpoint_voltage_mv: np.ndarray,
) -> np.ndarray:
    """Invert the same local exponential step to obtain spike event time."""
    initial = np.asarray(initial_voltage_mv, dtype=float)
    rhs = np.asarray(rhs_per_ms, dtype=float)
    jacobian = np.asarray(effective_jacobian_per_ms, dtype=float)
    interval = np.asarray(active_interval_ms, dtype=float)
    start_fraction = np.asarray(active_start_fraction, dtype=float)
    endpoint = np.asarray(endpoint_voltage_mv, dtype=float)
    delta = float(threshold_mv) - initial
    linear_fraction = np.divide(
        delta,
        endpoint - initial,
        out=np.ones_like(delta),
        where=(endpoint - initial) > 0.0,
    )
    crossing_ms = np.clip(linear_fraction, 0.0, 1.0) * interval
    nonlinear = (
        (np.abs(jacobian) > 1.0e-10)
        & (rhs > 0.0)
        & (interval > 0.0)
    )
    log_argument = np.ones_like(delta)
    log_argument[nonlinear] = (
        1.0 + jacobian[nonlinear] * delta[nonlinear] / rhs[nonlinear]
    )
    valid = nonlinear & (log_argument > 0.0)
    crossing_ms[valid] = (
        np.log(log_argument[valid]) / jacobian[valid]
    )
    crossing_ms = np.clip(crossing_ms, 0.0, interval)
    return np.clip(
        start_fraction + crossing_ms / float(outer_dt_ms), 0.0, 1.0
    )


def advance_exponential_rate_hysteresis(
    initial_rate_hz: float,
    event_fractions: np.ndarray,
    event_jump_hz: float,
    dt_ms: float,
    tau_ms: float,
    on_threshold_hz: float,
    off_threshold_hz: float,
    armed: bool,
) -> Tuple[float, bool, np.ndarray]:
    """Advance an exponential spike-rate kernel and exact event hysteresis."""
    fractions = np.sort(np.asarray(event_fractions, dtype=float))
    if np.any(~np.isfinite(fractions)) or np.any(
        (fractions < 0.0) | (fractions > 1.0)
    ):
        raise ValueError("event_fractions must be finite and in [0, 1]")
    if not (
        math.isfinite(initial_rate_hz)
        and initial_rate_hz >= 0.0
        and math.isfinite(event_jump_hz)
        and event_jump_hz > 0.0
        and math.isfinite(dt_ms)
        and dt_ms > 0.0
        and math.isfinite(tau_ms)
        and tau_ms > 0.0
        and 0.0 <= off_threshold_hz < on_threshold_hz
    ):
        raise ValueError("invalid exponential-rate hysteresis parameters")
    current_rate = float(initial_rate_hz)
    current_fraction = 0.0
    is_armed = bool(armed or current_rate <= off_threshold_hz)
    onset_fractions: List[float] = []
    for event_fraction in fractions:
        decayed_rate = current_rate * math.exp(
            -(float(event_fraction) - current_fraction) * dt_ms / tau_ms
        )
        if (
            not is_armed
            and current_rate > off_threshold_hz >= decayed_rate
        ):
            is_armed = True
        current_rate = decayed_rate
        if (
            is_armed
            and current_rate < on_threshold_hz
            <= current_rate + event_jump_hz
        ):
            onset_fractions.append(float(event_fraction))
            is_armed = False
        current_rate += event_jump_hz
        current_fraction = float(event_fraction)
    endpoint_rate = current_rate * math.exp(
        -(1.0 - current_fraction) * dt_ms / tau_ms
    )
    if (
        not is_armed
        and current_rate > off_threshold_hz >= endpoint_rate
    ):
        is_armed = True
    return endpoint_rate, is_armed, np.asarray(onset_fractions, dtype=float)


def allocate_fractional_labels(
    n: int, fractions: Sequence[float], rng: np.random.Generator
) -> np.ndarray:
    """Allocate reproducible phenotype counts by largest remainder.

    When the population is at least as large as the number of non-zero
    phenotypes, every supported phenotype is represented at least once. This
    avoids silently losing a declared subtype through random finite sampling.
    """
    fraction_array = np.asarray(fractions, dtype=float)
    raw = n * fraction_array
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts))[: n - int(np.sum(counts))]:
        counts[index] += 1
    nonzero = np.flatnonzero(fraction_array > 0.0)
    if n >= len(nonzero):
        for missing in nonzero[counts[nonzero] == 0]:
            donor_order = np.argsort(-counts)
            donor = next(
                int(index) for index in donor_order
                if counts[index] > 1
            )
            counts[donor] -= 1
            counts[missing] += 1
    labels = np.repeat(np.arange(len(fraction_array), dtype=np.int8), counts)
    rng.shuffle(labels)
    return labels


def allocate_grouped_fractional_labels(
    group_sizes: Sequence[int],
    fractions: Sequence[float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, ...]:
    """Allocate one global exact phenotype count across local groups.

    Source fractions are first rounded once over the whole sampled class.
    Rare phenotypes are then distributed as evenly as integer capacities
    permit, with outcome-blind randomized tie breaking and within-group order.
    This preserves global prevalence without forcing every small context to
    contain every phenotype.
    """
    sizes = np.asarray(group_sizes, dtype=int)
    if sizes.ndim != 1 or len(sizes) == 0 or np.any(sizes <= 0):
        raise ValueError("group_sizes must be a nonempty positive integer vector")
    total = int(np.sum(sizes))
    global_labels = allocate_fractional_labels(total, fractions, rng)
    global_counts = np.bincount(
        global_labels, minlength=len(fractions)
    ).astype(int)
    allocation = np.zeros((len(sizes), len(fractions)), dtype=int)
    remaining = sizes.copy()
    # Rare-first allocation prevents a rare measured phenotype from being
    # consumed by the same local context merely because of iteration order.
    for label in np.argsort(global_counts, kind="stable"):
        for _ in range(int(global_counts[label])):
            eligible = np.flatnonzero(remaining > 0)
            normalized = allocation[eligible, label] / sizes[eligible]
            best = eligible[np.isclose(normalized, np.min(normalized))]
            if len(best) > 1:
                best_remaining = remaining[best]
                best = best[best_remaining == np.max(best_remaining)]
            group = int(rng.choice(best))
            allocation[group, label] += 1
            remaining[group] -= 1
    if np.any(remaining) or not np.array_equal(
        np.sum(allocation, axis=0), global_counts
    ):
        raise RuntimeError("grouped phenotype allocation failed exact counts")
    result: List[np.ndarray] = []
    for row in allocation:
        labels = np.repeat(
            np.arange(len(fractions), dtype=np.int8), row
        )
        rng.shuffle(labels)
        result.append(labels)
    return tuple(result)


def allocate_class_population_mask(
    neuron_population: np.ndarray,
    cell_class: str,
    positive_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Instantiate one exact-count sample-derived subset across the class.

    The source fraction is a class-sample prevalence, not a requirement that
    every side/phase pool contain a positive cell.  Sampling separately inside
    each small pool would systematically inflate rare phenotypes.  Selection
    is structural and outcome-blind; pair-level measurements are handled by a
    separate grouped edge sampler.
    """
    if cell_class not in CLASSES:
        raise ValueError(f"Unknown class for population mask: {cell_class}")
    if not 0.0 < positive_fraction < 1.0:
        raise ValueError("positive_fraction must be in (0, 1)")
    mask = np.zeros(len(neuron_population), dtype=bool)
    ids = np.flatnonzero(np.asarray([
        POPULATIONS[population].rsplit("_", 2)[0] == cell_class
        for population in neuron_population
    ]))
    if len(ids) < 2:
        raise ValueError(f"{cell_class} requires at least two cells for a subset")
    n_selected = max(1, min(
        len(ids) - 1,
        int(math.floor(positive_fraction * len(ids) + 0.5)),
    ))
    selected = np.sort(rng.choice(ids, size=n_selected, replace=False))
    mask[selected] = True
    return mask


def sample_symmetric_pair_edges(
    neuron_ids: np.ndarray,
    pair_probability: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample an exact-count undirected local-pair graph, without self edges."""
    ids = np.asarray(neuron_ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("neuron_ids must be a one-dimensional unique array")
    if not 0.0 < pair_probability < 1.0:
        raise ValueError("pair_probability must be in (0, 1)")
    if len(ids) < 2:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    local_i, local_j = np.triu_indices(len(ids), k=1)
    n_possible = len(local_i)
    n_selected = max(1, int(math.floor(
        pair_probability * n_possible + 0.5
    )))
    chosen = np.sort(rng.choice(n_possible, size=n_selected, replace=False))
    return ids[local_i[chosen]], ids[local_j[chosen]]


def sample_grouped_symmetric_pair_edges(
    neuron_id_groups: Sequence[np.ndarray],
    pair_probability: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Sample one exact count from the union of local candidate pairs.

    Each candidate remains inside its supplied side/phase group, but rounding
    occurs once across all groups.  This prevents small pools from turning a
    measured pair incidence into one mandatory edge per pool (for example the
    default two-cell phasic V2a pools would otherwise yield 100% coupling).
    """
    if not 0.0 < pair_probability < 1.0:
        raise ValueError("pair_probability must be in (0, 1)")
    candidate_sources: List[np.ndarray] = []
    candidate_targets: List[np.ndarray] = []
    for neuron_ids in neuron_id_groups:
        ids = np.asarray(neuron_ids, dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise ValueError("each neuron-id group must be one-dimensional and unique")
        if len(ids) < 2:
            continue
        local_i, local_j = np.triu_indices(len(ids), k=1)
        candidate_sources.append(ids[local_i])
        candidate_targets.append(ids[local_j])
    if not candidate_sources:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty, 0
    sources = np.concatenate(candidate_sources)
    targets = np.concatenate(candidate_targets)
    n_possible = len(sources)
    n_selected = max(1, int(math.floor(
        pair_probability * n_possible + 0.5
    )))
    chosen = np.sort(rng.choice(n_possible, size=n_selected, replace=False))
    return sources[chosen], targets[chosen], n_possible


def symmetric_pair_gap_current(
    voltage_mv: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    conductance_ns: float,
) -> np.ndarray:
    """Return equal-and-opposite current for every undirected gap edge."""
    current = np.zeros_like(voltage_mv)
    if len(source) == 0:
        return current
    pair_current = conductance_ns * (
        voltage_mv[target] - voltage_mv[source]
    )
    np.add.at(current, source, pair_current)
    np.add.at(current, target, -pair_current)
    return current


CONFIG_DOMAIN_POLICY_GROUPS: Dict[str, Tuple[str, ...]] = {
    "positive_integer": (
        "rg_neurons", "relay_neurons", "pf_neurons", "mn_neurons",
        "long_n_epochs", "long_baseline_end_epoch", "long_demand_start_epoch",
        "long_challenge_epoch", "long_demand_end_epoch",
        "recovery_consecutive_cycles",
    ),
    "strictly_positive": (
        "dt_ms", "duration_s", "slope_factor_mv", "refractory_ms",
        "nap_activation_slope_mv", "nap_inactivation_slope_mv",
        "nap_inactivation_tau_ms", "rg_ltype_ca_pic_activation_slope_mv",
        "rg_ltype_ca_pic_activation_tau_ms", "rg_h_slope_mv", "rg_h_tau_ms",
        "rg_t_activation_slope_mv", "rg_t_inactivation_slope_mv",
        "rg_t_inactivation_tau_ms", "rg_a_activation_slope_mv",
        "rg_a_inactivation_slope_mv", "rg_a_inactivation_tau_ms",
        "m_activation_slope_mv",
        "m_activation_tau_ms", "rg_m_conductance_scale", "calcium_decay_ms",
        "calcium_half_activation", "calcium_hill_coefficient",
        "pf_slow_integration_half_pa", "pf_slow_integration_rise_ms",
        "pf_slow_integration_decay_ms", "v2a_delayed_activation_slope_mv",
        "v2a_delayed_relief_tau_ms", "v2a_phasic_adaptation_multiplier",
        "v2a_h_slope_mv", "v2a_h_tau_ms",
        "v3_h_slope_mv", "v3_h_tau_ms", "v3_t_activation_slope_mv",
        "v3_t_inactivation_slope_mv", "v3_t_inactivation_tau_ms",
        "v3_dorsal_h_conductance_multiplier",
        "v3_dorsal_adaptation_multiplier", "renshaw_nachr_decay_ms",
        "renshaw_glutamate_decay_ms",
        "renshaw_h_slope_mv", "renshaw_h_tau_ms",
        "renshaw_calcium_decay_ms", "renshaw_calcium_half_activation",
        "mn_dendrite_capacitance_pf", "mn_dendrite_leak_ns",
        "mn_nap_pic_activation_slope_mv", "mn_nap_pic_inactivation_slope_mv",
        "mn_nap_pic_inactivation_tau_ms", "mn_ltype_ca_pic_activation_slope_mv",
        "mn_ltype_ca_pic_activation_tau_ms", "mn_calcium_decay_ms",
        "mn_calcium_half_activation", "noise_tau_ms", "noise_burst_multiplier",
        "phase_kick_current_pa", "phase_kick_duration_ms",
        "excitatory_pulse_current_pa", "pulse_duration_ms",
        "excitatory_synapse_decay_ms", "inhibitory_synapse_decay_ms",
        "rg_recurrent_excitation_pa", "rg_to_v1_pa", "v1_to_antagonist_rg_pa",
        "rg_to_v2a_pa", "rg_to_v0d_pa", "v2a_to_v0v_pa", "rg_to_v3_pa",
        "v0d_cross_inhibition_pa", "v0v_to_cross_v1_pa",
        "v3_to_contralateral_mn_pa", "v3_vlat_to_ipsilateral_mn_pa",
        "mn_to_v3_vlat_glutamate_pa", "rg_to_pf_pa", "pf_recurrent_excitation_pa",
        "v1_to_antagonist_pf_pa", "rg_to_v2b_pa",
        "v2b_to_antagonist_rg_pa", "v2b_to_antagonist_pf_pa", "pf_to_mn_pa",
        "v1_to_antagonist_mn_pa", "v2b_to_antagonist_mn_pa",
        "mn_to_v1ren_nachr_pa", "mn_to_v1ren_glutamate_pa",
        "v1ren_to_mn_pa", "v1ren_to_v1ia_pa",
        "v2a_to_pf_pa", "v2a_to_mn_pa", "mt_activity_tau_ms",
        "mt_track_decay_ms", "mt_track_max", "vesicle_fast_recovery_ms",
        "vesicle_slow_recovery_ms", "slow_replenishment_resource_recovery_ms",
        "nmj_release_delay_ms", "nmj_vesicle_recovery_ms", "nmj_ach_decay_ms",
        "nmj_ach_release_gain", "nmj_endplate_gain_mv", "nmj_endplate_tau_ms",
        "nmj_nachr_half_ach", "nmj_nachr_hill", "muscle_fiber_slope_mv",
        "muscle_calcium_tau_ms", "muscle_effective_sr_release_per_ms",
        "muscle_calcium_half", "muscle_calcium_hill",
        "muscle_activation_tau_ms", "joint_inertia", "joint_damping",
        "joint_stiffness", "muscle_torque_gain", "muscle_length_scale",
        "ia_length_gain", "ia_velocity_gain", "ib_force_gain", "ia_to_pf_pa",
        "ia_to_mn_pa", "ia_to_v1ia_pa", "ib_effective_spinal_to_pf_pa",
        "ib_effective_spinal_to_mn_pa",
        "sensory_resource_recovery_ms", "long_epoch_duration_s", "rate_tau_ms",
        "minimum_interburst_s", "rg_pf_match_window_s",
        "rg_mn_match_post_window_s",
        "rg_nap_conductance_ns", "rg_ltype_ca_pic_conductance_ns",
        "rg_h_conductance_ns", "rg_t_conductance_ns",
        "rg_a_conductance_ns", "rg_gap_conductance_ns",
        "rg_m_conductance_ns", "kca_conductance_ns",
        "calcium_spike_increment", "pf_slow_integration_conductance_ns",
        "v2a_delayed_onset_conductance_ns", "v2a_gap_conductance_ns",
        "v2a_h_conductance_ns",
        "v3_ventral_h_conductance_ns", "v3_t_conductance_ns",
        "renshaw_h_conductance_ns", "renshaw_sk_conductance_ns",
        "renshaw_calcium_spike_increment", "mn_soma_dendrite_coupling_ns",
        "mn_nap_pic_conductance_ns", "mn_ltype_ca_pic_conductance_ns",
        "mn_ahp_conductance_ns",
        "mn_calcium_spike_increment", "mt_activity_spike_increment",
        "mt_nucleation_per_ms", "mt_slow_replenishment_gain",
        "sensory_resource_depletion_per_s",
    ),
    "nonnegative": (
        "independent_noise_sigma_pa", "population_common_noise_sigma_pa",
        "ia_tonic", "ib_tonic",
        "rg_mn_match_pre_window_s",
    ),
    "closed_fraction": (
        "static_kca_activation_reference",
        "mn_dendritic_synaptic_fraction", "mt_impaired_nucleation_scale",
        "long_rrp_challenge_floor",
        "long_replenishment_resource_challenge_floor",
    ),
    "open_fraction": (
        "excitatory_pulse_cycle_fraction", "inhibitory_pulse_cycle_fraction",
        "v2b_flexor_target_relative_gain", "v3_dorsal_fraction",
        "v3_vlat_fraction_of_ventral",
        "recovery_frequency_tolerance_fraction", "rg_pic_positive_fraction",
        "rg_m_positive_fraction", "rg_kca_positive_fraction",
        "rg_h_positive_fraction", "rg_t_positive_fraction",
        "rg_a_positive_fraction", "rg_gap_pair_probability",
        "v2a_tonic_gap_pair_probability", "v2a_phasic_gap_pair_probability",
    ),
    "positive_closed_fraction": (
        "recurrent_connection_probability", "local_connection_probability",
        "commissural_connection_probability", "mt_impaired_lifetime_scale",
        "vesicle_depletion_fraction", "challenge_route_fraction",
        "nmj_release_probability",
    ),
    "bounded_heterogeneity": (
        "drive_heterogeneity_fraction", "synaptic_heterogeneity_fraction",
    ),
    "at_least_one": (
        "load_unilateral_resistance_multiplier",
        "load_bilateral_high_resistance_multiplier", "extensor_force_scale_prior",
        "ib_effective_extensor_context_gain",
    ),
    "strictly_negative": ("inhibitory_pulse_current_pa", "pf_deletion_current_pa"),
    "nonnegative_schedule": (
        "burn_in_s", "perturbation_start_s", "perturbation_end_s",
        "pulse_arm_after_s",
    ),
    "finite_voltage": (
        "leak_reversal_mv", "threshold_mv", "reset_mv", "spike_peak_mv",
        "sodium_reversal_mv", "nap_activation_half_mv",
        "nap_inactivation_half_mv", "rg_ltype_ca_pic_activation_half_mv",
        "rg_ltype_ca_reversal_mv", "rg_h_reversal_mv", "rg_h_half_mv",
        "rg_t_reversal_mv", "rg_t_activation_half_mv",
        "rg_t_inactivation_half_mv", "rg_a_activation_half_mv",
        "rg_a_inactivation_half_mv", "m_activation_half_mv",
        "potassium_reversal_mv", "excitatory_reversal_mv",
        "v2a_delayed_activation_half_mv", "v2a_h_reversal_mv",
        "v2a_h_half_mv", "v3_h_reversal_mv", "v3_h_half_mv",
        "v3_t_reversal_mv", "v3_t_activation_half_mv",
        "v3_t_inactivation_half_mv", "renshaw_h_reversal_mv",
        "renshaw_h_half_mv", "mn_nap_pic_activation_half_mv",
        "mn_nap_pic_inactivation_half_mv", "mn_ltype_ca_pic_activation_half_mv",
        "mn_ltype_ca_reversal_mv", "inhibitory_reversal_mv",
        "synaptic_reference_voltage_mv", "muscle_fiber_rest_mv",
        "muscle_fiber_threshold_mv",
    ),
    "burst_threshold": (
        "burst_on_threshold_hz", "burst_off_threshold_hz",
        "pf_burst_on_threshold_hz", "pf_burst_off_threshold_hz",
        "mn_burst_on_threshold_hz", "mn_burst_off_threshold_hz",
    ),
    "phase_angle": ("phase_slip_threshold_deg",),
    "speed_offset_tuple": ("descending_rg_drive_offsets_pa",),
    "subtype_fraction_tuple": ("v2a_variant_fractions",),
    "subtype_probability_tuple": ("v2a_h_positive_fractions",),
    "delay_tuple": (
        "recurrent_delay_bins_ms", "local_delay_bins_ms",
        "commissural_delay_bins_ms",
    ),
    "static_support_tuple": ("mt_static_route_supports",),
    "track_initial_relative": ("mt_initial_track_fraction",),
}


def _build_config_domain_policy() -> Dict[str, str]:
    policy: Dict[str, str] = {}
    for policy_name, names in CONFIG_DOMAIN_POLICY_GROUPS.items():
        for name in names:
            if name in policy:
                raise RuntimeError(
                    f"Config domain policy overlap for {name}: "
                    f"{policy[name]} and {policy_name}"
                )
            policy[name] = policy_name
    declared = {item.name for item in fields(Config)}
    missing = declared - set(policy)
    extra = set(policy) - declared
    if missing or extra:
        raise RuntimeError(
            f"Config domain policy coverage mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return policy


CONFIG_DOMAIN_POLICY = _build_config_domain_policy()


def validate_config(cfg: Config, protocol: str | None = None) -> None:
    """Fail closed: every Config field has exactly one explicit domain policy."""
    validate_cell_class_equations()
    validate_biological_interface_equations()
    validate_literature_evidence_contract()
    if set(CONFIG_DOMAIN_POLICY) != {item.name for item in fields(cfg)}:
        raise RuntimeError("Config domain policy no longer exactly covers Config")
    for item in fields(cfg):
        value = getattr(cfg, item.name)
        values = value if isinstance(value, tuple) else (value,)
        for scalar in values:
            if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
                raise TypeError(f"Config.{item.name} must be numeric")
            if not math.isfinite(float(scalar)):
                raise ValueError(f"Config.{item.name} must be finite")
        policy = CONFIG_DOMAIN_POLICY[item.name]
        if policy == "positive_integer":
            if isinstance(value, bool) or int(value) != value:
                raise TypeError(f"Config.{item.name} must be an integer")
            if value <= 0:
                raise ValueError(f"Config.{item.name} must be positive")
        elif policy in {"strictly_positive", "burst_threshold"} and value <= 0:
            raise ValueError(f"Config.{item.name} must be positive")
        elif policy in {"nonnegative", "nonnegative_schedule"} and value < 0:
            raise ValueError(f"Config.{item.name} must be nonnegative")
        elif policy == "closed_fraction" and not 0.0 <= value <= 1.0:
            raise ValueError(f"Config.{item.name} must be in [0, 1]")
        elif policy == "open_fraction" and not 0.0 < value < 1.0:
            raise ValueError(f"Config.{item.name} must be in (0, 1)")
        elif policy == "positive_closed_fraction" and not 0.0 < value <= 1.0:
            raise ValueError(f"Config.{item.name} must be in (0, 1]")
        elif policy == "bounded_heterogeneity" and not 0.0 <= value <= 0.5:
            raise ValueError(f"Config.{item.name} must be in [0, 0.5]")
        elif policy == "at_least_one" and value < 1.0:
            raise ValueError(f"Config.{item.name} must be at least one")
        elif policy == "strictly_negative" and value >= 0.0:
            raise ValueError(f"Config.{item.name} must be negative")
        elif policy == "phase_angle" and not 0.0 < value <= 180.0:
            raise ValueError(f"Config.{item.name} must be in (0, 180]")

    if not 0.0 <= cfg.burn_in_s < cfg.duration_s:
        raise ValueError("Config must satisfy 0 <= burn_in_s < duration_s")
    duration_steps = cfg.duration_s * 1000.0 / cfg.dt_ms
    if duration_steps < 1.0 or not math.isclose(
        duration_steps, round(duration_steps), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("duration_s must contain a positive integer number of dt_ms steps")
    steps_per_public_ms = 1.0 / cfg.dt_ms
    if not math.isclose(
        steps_per_public_ms,
        round(steps_per_public_ms),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("dt_ms must divide the public 1-ms sample interval exactly")
    if cfg.rg_neurons < 2 or cfg.pf_neurons < 2:
        raise ValueError("recurrent RG and PF populations require at least two neurons")
    if cfg.relay_neurons < 3:
        raise ValueError(
            "relay_neurons must be at least three so every declared V2a phenotype exists"
        )
    if not cfg.perturbation_start_s < cfg.perturbation_end_s:
        raise ValueError("perturbation_start_s must precede perturbation_end_s")
    if cfg.phase_kick_duration_ms > 1000.0 * (
        cfg.perturbation_end_s - cfg.perturbation_start_s
    ):
        raise ValueError("phase kick must fit inside the declared perturbation interval")
    if min(cfg.phase_kick_duration_ms, cfg.pulse_duration_ms) < cfg.dt_ms:
        raise ValueError("pulse and phase-kick durations must span at least one step")
    if cfg.v2a_phasic_adaptation_multiplier <= 1.0:
        raise ValueError("V2a phasic adaptation multiplier must exceed one")
    if cfg.v3_dorsal_adaptation_multiplier <= 1.0:
        raise ValueError("V3 dorsal adaptation multiplier must exceed one")
    if cfg.v3_dorsal_h_conductance_multiplier <= 1.0:
        raise ValueError("V3 dorsal Ih conductance multiplier must exceed one")
    if not (
        cfg.burst_on_threshold_hz > cfg.burst_off_threshold_hz
        and cfg.pf_burst_on_threshold_hz > cfg.pf_burst_off_threshold_hz
        and cfg.mn_burst_on_threshold_hz > cfg.mn_burst_off_threshold_hz
    ):
        raise ValueError("Every burst on-threshold must exceed its off-threshold")
    if not (
        cfg.potassium_reversal_mv < cfg.inhibitory_reversal_mv
        < cfg.leak_reversal_mv < cfg.threshold_mv < cfg.spike_peak_mv
        < cfg.sodium_reversal_mv
    ):
        raise ValueError("membrane voltage landmarks have an invalid order")
    if not cfg.reset_mv < cfg.threshold_mv:
        raise ValueError("reset_mv must be below threshold_mv")
    if not (
        cfg.excitatory_reversal_mv > cfg.synaptic_reference_voltage_mv
        > cfg.inhibitory_reversal_mv
    ):
        raise ValueError("synaptic reference voltage must lie between receptor reversals")
    if not cfg.threshold_mv < cfg.excitatory_reversal_mv < cfg.sodium_reversal_mv:
        raise ValueError("excitatory reversal must lie between threshold and sodium reversal")
    if cfg.mn_ltype_ca_reversal_mv <= cfg.spike_peak_mv:
        raise ValueError("MN L-type Ca reversal must exceed spike_peak_mv")
    if cfg.rg_ltype_ca_reversal_mv <= cfg.spike_peak_mv:
        raise ValueError("RG L-type Ca reversal must exceed spike_peak_mv")
    if cfg.rg_t_reversal_mv <= cfg.spike_peak_mv:
        raise ValueError("RG T-type Ca reversal must exceed spike_peak_mv")
    if not (
        cfg.muscle_fiber_rest_mv < cfg.muscle_fiber_threshold_mv
        < cfg.muscle_fiber_rest_mv + cfg.nmj_endplate_gain_mv
    ):
        raise ValueError("muscle-fiber threshold must be reachable from rest by the endplate gain")
    if len(cfg.descending_rg_drive_offsets_pa) != len(SPEED_LEVELS):
        raise ValueError("descending_rg_drive_offsets_pa must have length three")
    if not (
        cfg.descending_rg_drive_offsets_pa[0]
        < cfg.descending_rg_drive_offsets_pa[1]
        < cfg.descending_rg_drive_offsets_pa[2]
        and cfg.descending_rg_drive_offsets_pa[1] == 0.0
    ):
        raise ValueError("speed offsets must increase strictly with an exact zero midpoint")
    if (
        len(cfg.v2a_variant_fractions) != 3
        or any(value <= 0.0 for value in cfg.v2a_variant_fractions)
        or not math.isclose(
            sum(cfg.v2a_variant_fractions), 1.0,
            rel_tol=0.0, abs_tol=1e-12,
        )
    ):
        raise ValueError("v2a_variant_fractions must be three positive values summing to one")
    if (
        len(cfg.v2a_h_positive_fractions) != 3
        or any(
            value <= 0.0 or value > 1.0
            for value in cfg.v2a_h_positive_fractions
        )
    ):
        raise ValueError(
            "v2a_h_positive_fractions must be three values in (0, 1]"
        )
    for name in (
        "recurrent_delay_bins_ms", "local_delay_bins_ms",
        "commissural_delay_bins_ms",
    ):
        values = getattr(cfg, name)
        if not values or any(value <= 0.0 for value in values) or any(
            right <= left for left, right in zip(values[:-1], values[1:])
        ):
            raise ValueError(f"Config.{name} must be nonempty, positive and strictly ascending")
    if not 0.0 <= cfg.mt_initial_track_fraction <= cfg.mt_track_max:
        raise ValueError("mt_initial_track_fraction must be in [0, mt_track_max]")
    if len(cfg.mt_static_route_supports) != len(MT_ROUTES) or any(
        not 0.0 < value <= cfg.mt_track_max
        for value in cfg.mt_static_route_supports
    ):
        raise ValueError("mt_static_route_supports must have ten values in (0, mt_track_max]")
    if not (
        1 <= cfg.long_baseline_end_epoch < cfg.long_demand_start_epoch
        < cfg.long_challenge_epoch <= cfg.long_demand_end_epoch
        < cfg.long_n_epochs
    ):
        raise ValueError("Long-protocol order must be baseline < demand < challenge < recovery")
    if (
        cfg.long_rrp_challenge_floor
        > cfg.long_replenishment_resource_challenge_floor
    ):
        raise ValueError(
            "RRP challenge floor may not exceed slow-resource floor"
        )
    perturbation_protocols = {
        "noise_burst", "speed_step", "phase_kick", "pf_deletion",
    }
    if protocol in perturbation_protocols and not (
        0.0 <= cfg.perturbation_start_s < cfg.perturbation_end_s
        <= cfg.duration_s
    ):
        raise ValueError(
            "active perturbation protocols require start < end <= duration"
        )
    if protocol == "pulse" and not (
        0.0 <= cfg.pulse_arm_after_s < cfg.duration_s
        and cfg.pulse_arm_after_s + cfg.pulse_duration_ms / 1000.0
        <= cfg.duration_s
    ):
        raise ValueError("pulse protocol must leave a complete post-arm pulse window")
    if protocol == "long" and not math.isclose(
        cfg.duration_s,
        cfg.long_n_epochs * cfg.long_epoch_duration_s,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "long duration must equal long_n_epochs * long_epoch_duration_s"
        )


def population_metadata(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sizes = np.asarray([
        cfg.rg_neurons if name.startswith("RG_")
        else cfg.pf_neurons if name.startswith("PF_")
        else cfg.mn_neurons if name.startswith("MN_")
        else cfg.relay_neurons
        for name in POPULATIONS
    ], dtype=int)
    neuron_population = np.repeat(np.arange(len(POPULATIONS)), sizes)
    neuron_local_index = np.concatenate([np.arange(size) for size in sizes])
    # MT eligibility is not selected by an expected coordination function.
    # Every modeled central presynaptic population receives local terminal
    # states; the NMJ and phenomenological sensory transducers are separate.
    mt_eligible = np.ones(len(POPULATIONS), dtype=float)
    return sizes, neuron_population, neuron_local_index, mt_eligible


def pathway_specs(cfg: Config) -> List[Pathway]:
    """Population scaffold; projection roles never masquerade as subclasses."""
    pathways: List[Pathway] = []
    for side, phase in SIDE_PHASES:
        opposite_phase = other_phase(phase)
        contralateral = other_side(side)
        rg = pop("RG", side, phase)
        pf = pop("PF", side, phase)
        v1ia = pop("V1Ia", side, phase)
        v1ren = pop("V1Ren", side, phase)
        v2b = pop("V2b", side, phase)
        v2a = pop("V2a", side, phase)
        v0d = pop("V0D", side, phase)
        v0v = pop("V0V", side, phase)
        v3 = pop("V3", side, phase)
        mn = pop("MN", side, phase)

        pathways.extend([
            Pathway("RG_recurrent", rg, rg, cfg.rg_recurrent_excitation_pa,
                    cfg.recurrent_connection_probability, cfg.recurrent_delay_bins_ms,
                    "rhythm_core"),
            Pathway("RG_to_V1Ia", rg, v1ia, cfg.rg_to_v1_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "fe_coordination"),
            Pathway("V1Ia_to_antagonist_RG", v1ia, pop("RG", side, opposite_phase),
                    -0.5 * cfg.v1_to_antagonist_rg_pa, cfg.local_connection_probability,
                    cfg.local_delay_bins_ms, "fe_coordination"),
            Pathway("RG_to_V2b", rg, v2b, cfg.rg_to_v2b_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "fe_coordination"),
            Pathway("V2b_to_antagonist_RG", v2b, pop("RG", side, opposite_phase),
                    -cfg.v2b_to_antagonist_rg_pa, cfg.local_connection_probability,
                    cfg.local_delay_bins_ms, "fe_coordination"),
            Pathway("RG_to_PF", rg, pf, cfg.rg_to_pf_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "pattern_transfer"),
            Pathway("PF_recurrent", pf, pf, cfg.pf_recurrent_excitation_pa,
                    cfg.recurrent_connection_probability, cfg.recurrent_delay_bins_ms,
                    "pattern_transfer"),
            Pathway("V1Ia_to_antagonist_PF", v1ia, pop("PF", side, opposite_phase),
                    -cfg.v1_to_antagonist_pf_pa, cfg.local_connection_probability,
                    cfg.local_delay_bins_ms, "pattern_transfer"),
            Pathway("V2b_to_antagonist_PF", v2b, pop("PF", side, opposite_phase),
                    -cfg.v2b_to_antagonist_pf_pa, cfg.local_connection_probability,
                    cfg.local_delay_bins_ms, "pattern_transfer"),
            Pathway("PF_to_MN", pf, mn, cfg.pf_to_mn_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "motor_output"),
            Pathway("V1Ia_to_antagonist_MN", v1ia, pop("MN", side, opposite_phase),
                    -cfg.v1_to_antagonist_mn_pa, cfg.local_connection_probability,
                    cfg.local_delay_bins_ms, "motor_output"),
            Pathway("V2b_to_antagonist_MN", v2b, pop("MN", side, opposite_phase),
                    -cfg.v2b_to_antagonist_mn_pa * (
                        1.0 if opposite_phase == "E"
                        else cfg.v2b_flexor_target_relative_gain
                    ), cfg.local_connection_probability,
                    cfg.local_delay_bins_ms, "premotor_inhibition"),
            Pathway("MN_to_V1Ren_nAChR", mn, v1ren,
                    cfg.mn_to_v1ren_nachr_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "renshaw_feedback",
                    topology_group="MN_to_V1Ren_pair"),
            Pathway("MN_to_V1Ren_GluR", mn, v1ren,
                    cfg.mn_to_v1ren_glutamate_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "renshaw_feedback",
                    topology_group="MN_to_V1Ren_pair"),
            Pathway("MN_to_V3_VLat_GluR", mn, v3,
                    cfg.mn_to_v3_vlat_glutamate_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "V3_ventral_recurrent_motor_microcircuit",
                    target_subphenotype="vlat"),
            Pathway("V1Ren_to_MN", v1ren, mn, -cfg.v1ren_to_mn_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "renshaw_feedback"),
            Pathway("V1Ren_to_V1Ia", v1ren, v1ia, -cfg.v1ren_to_v1ia_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "renshaw_feedback"),
            Pathway("RG_to_V2a", rg, v2a, cfg.rg_to_v2a_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "V2a_recruitment"),
            Pathway("V2a_to_V0V", v2a, v0v, cfg.v2a_to_v0v_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "commissural"),
            Pathway("V2a_to_PF", v2a, pf, cfg.v2a_to_pf_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "ipsilateral_relay"),
            Pathway("V2a_to_MN", v2a, mn, cfg.v2a_to_mn_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "ipsilateral_relay"),
            Pathway("V0V_to_cross_V1Ia", v0v, pop("V1Ia", contralateral, opposite_phase),
                    cfg.v0v_to_cross_v1_pa, cfg.commissural_connection_probability,
                    cfg.commissural_delay_bins_ms, "commissural"),
            Pathway("RG_to_V0D", rg, v0d, cfg.rg_to_v0d_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "commissural"),
            Pathway("V0D_cross_inhibition", v0d, pop("RG", contralateral, phase),
                    -cfg.v0d_cross_inhibition_pa, cfg.commissural_connection_probability,
                    cfg.commissural_delay_bins_ms, "commissural"),
            Pathway("RG_to_V3", rg, v3, cfg.rg_to_v3_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "bilateral_balance"),
            Pathway("V3_VLat_to_ipsilateral_MN", v3, mn,
                    cfg.v3_vlat_to_ipsilateral_mn_pa,
                    cfg.local_connection_probability, cfg.local_delay_bins_ms,
                    "V3_ventrolateral_motor_microcircuit",
                    source_subphenotype="vlat"),
        ])
        pathways.append(Pathway(
            (
                "V3_ventral_to_contralateral_MN_flexor"
                if phase == "F" else "V3_ventral_to_contralateral_MN_extensor"
            ),
            v3, pop("MN", contralateral, phase),
            cfg.v3_to_contralateral_mn_pa,
            cfg.commissural_connection_probability,
            cfg.commissural_delay_bins_ms, "motor_output",
            source_subphenotype="ventral",
        ))

    direct = {
        "V1Ia_to_antagonist_MN": "reciprocal Ia inhibitory action is directly established",
        "MN_to_V1Ren_nAChR": "effective cholinergic component in a directly supported mixed MN-to-Renshaw pair; separate component edge state is H-level",
        "MN_to_V1Ren_GluR": "effective glutamatergic component in a directly supported mixed MN-to-Renshaw pair; separate component edge state is H-level",
        "V1Ren_to_MN": "recurrent Renshaw inhibition of motor neurons is directly established",
        "V1Ren_to_V1Ia": "Renshaw inhibition of Ia inhibitory interneurons is experimentally supported",
        "V2a_to_V0V": "direct excitatory V2a contacts onto molecular V0 commissural neurons are experimentally supported",
        "V2a_to_MN": "population-level ipsilateral V2a-to-MN output is experimentally supported; firing-phenotype contribution is unresolved",
        "V3_VLat_to_ipsilateral_MN": "ventrolateral V3 excitation of ipsilateral motoneurons is directly supported",
        "MN_to_V3_VLat_GluR": "recurrent glutamatergic MN excitation of ventral/ventrolateral V3 is directly supported",
    }
    consensus = {
        "V2b_to_antagonist_MN": "V2b contribution to flexor-extensor motor inhibition is literature supported",
        "V3_ventral_to_contralateral_MN_flexor": "ventral V3 contribution to predominantly contralateral flexor motor-pool input is anatomically/transsynaptically supported",
        "V3_ventral_to_contralateral_MN_extensor": "ventral V3 contribution to predominantly contralateral extensor motor-pool input is anatomically/transsynaptically supported",
    }
    classified: List[Pathway] = []
    for spec in pathways:
        source_class = POPULATIONS[spec.source_population].rsplit("_", 2)[0]
        if spec.name in direct:
            classified.append(replace(
                spec, evidence_class="direct_experimental",
                evidence_note=direct[spec.name],
                mt_route=source_class, recruitment_axis="none",
            ))
        elif spec.name in consensus:
            classified.append(replace(
                spec, evidence_class="literature_consensus",
                evidence_note=consensus[spec.name],
                mt_route=source_class, recruitment_axis="none",
            ))
        else:
            classified.append(replace(
                spec, evidence_class="model_hypothesis",
                evidence_note=(
                    "functional coarse-graining or exact gain/target assignment; "
                    "requires sensitivity analysis"
                ),
                mt_route=source_class, recruitment_axis="none",
            ))
    return classified


def validate_pathway_class_coverage(
    cfg: Config,
    specs: Sequence[Pathway] | None = None,
) -> None:
    """Validate exact typed topology, sign, probability and physical delay.

    ``specs`` is injectable solely so mutation tests can prove fail-closed
    behavior. Normal execution validates the list returned by
    :func:`pathway_specs`.
    """
    validate_class_execution_contracts()
    checked = list(pathway_specs(cfg) if specs is None else specs)
    if not checked:
        raise ValueError("Pathway scaffold may not be empty")
    allowed_subphenotypes = {"all", "ventral", "vlat"}
    expected_source_subphenotypes = {
        "V3_ventral_to_contralateral_MN_flexor": "ventral",
        "V3_ventral_to_contralateral_MN_extensor": "ventral",
        "V3_VLat_to_ipsilateral_MN": "vlat",
    }
    expected_target_subphenotypes = {
        "MN_to_V3_VLat_GluR": "vlat",
    }
    for spec in checked:
        if not isinstance(spec.source_population, int) or not isinstance(
            spec.target_population, int
        ):
            raise TypeError(f"Pathway {spec.name} population ids must be integers")
        if not 0 <= spec.source_population < len(POPULATIONS) or not (
            0 <= spec.target_population < len(POPULATIONS)
        ):
            raise ValueError(f"Pathway {spec.name} references an unknown population")
        if not spec.name or not math.isfinite(spec.population_weight_pa) or (
            spec.population_weight_pa == 0.0
        ):
            raise ValueError(f"Pathway {spec.name!r} must have a finite nonzero weight")
        if isinstance(spec.connection_probability, bool) or not math.isfinite(
            spec.connection_probability
        ) or not (
            0.0 < spec.connection_probability <= 1.0
        ):
            raise ValueError(f"Pathway {spec.name} probability must be in (0, 1]")
        if not spec.delay_bins_ms or any(
            isinstance(value, bool) or not math.isfinite(value) or value <= 0.0
            for value in spec.delay_bins_ms
        ) or any(
            right <= left
            for left, right in zip(
                spec.delay_bins_ms[:-1], spec.delay_bins_ms[1:]
            )
        ):
            raise ValueError(f"Pathway {spec.name} has an invalid physical delay")
        source_class = POPULATIONS[spec.source_population].rsplit("_", 2)[0]
        target_class = POPULATIONS[spec.target_population].rsplit("_", 2)[0]
        if (
            spec.source_subphenotype not in allowed_subphenotypes
            or spec.target_subphenotype not in allowed_subphenotypes
        ):
            raise ValueError(f"Pathway {spec.name} has an unknown subphenotype selector")
        if spec.source_subphenotype != "all" and source_class != "V3":
            raise ValueError(f"Pathway {spec.name} has a non-V3 source selector")
        if spec.target_subphenotype != "all" and target_class != "V3":
            raise ValueError(f"Pathway {spec.name} has a non-V3 target selector")
        if (
            spec.source_subphenotype
            != expected_source_subphenotypes.get(spec.name, "all")
            or spec.target_subphenotype
            != expected_target_subphenotypes.get(spec.name, "all")
        ):
            raise ValueError(
                f"Pathway {spec.name} subphenotype selector contract drift"
            )
        contract = CLASS_EXECUTION_CONTRACTS[source_class]
        observed_sign = 1 if spec.population_weight_pa > 0.0 else -1
        if observed_sign != contract.output_weight_sign:
            raise ValueError(f"Pathway {spec.name} violates {source_class} output sign")
        if spec.mt_route != source_class:
            raise ValueError(
                f"MT route must equal presynaptic identity: {spec.name} "
                f"has {spec.mt_route}, expected {source_class}"
            )
        if spec.recruitment_axis != "none":
            raise ValueError(f"Outcome-loaded recruitment label retained: {spec.name}")

    evidence_classes_by_name: Dict[str, set[str]] = {}
    for spec in checked:
        evidence_classes_by_name.setdefault(spec.name, set()).add(
            spec.evidence_class
        )
    if set(evidence_classes_by_name) != set(PATHWAY_EVIDENCE):
        raise ValueError("runtime pathway/evidence binding names drifted")
    expected_mechanism_grade = {
        "direct_experimental": {"A"},
        "literature_consensus": {"B"},
        "model_hypothesis": {"C", "H"},
    }
    for pathway_name, observed_classes in evidence_classes_by_name.items():
        if len(observed_classes) != 1:
            raise ValueError(f"{pathway_name} has inconsistent evidence classes")
        observed_class = next(iter(observed_classes))
        if observed_class not in expected_mechanism_grade:
            raise ValueError(f"{pathway_name} has unknown evidence class")
        if PATHWAY_EVIDENCE[pathway_name].mechanism_evidence not in (
            expected_mechanism_grade[observed_class]
        ):
            raise ValueError(
                f"{pathway_name} runtime/source evidence classification drift"
            )

    paired_groups: Dict[str, List[Pathway]] = {}
    for spec in checked:
        if not spec.topology_group:
            raise ValueError(f"Pathway {spec.name} has an empty topology group")
        if spec.topology_group != "none":
            paired_groups.setdefault(spec.topology_group, []).append(spec)
    if set(paired_groups) != {"MN_to_V1Ren_pair"}:
        raise ValueError("paired-topology pathway contract drift")
    paired = paired_groups["MN_to_V1Ren_pair"]
    if {spec.name for spec in paired} != {
        "MN_to_V1Ren_nAChR", "MN_to_V1Ren_GluR",
    }:
        raise ValueError("MN-to-Renshaw paired child-path contract drift")
    paired_by_population: Dict[Tuple[int, int], List[Pathway]] = {}
    for spec in paired:
        paired_by_population.setdefault((
            spec.source_population, spec.target_population,
        ), []).append(spec)
    for population_pair, children in paired_by_population.items():
        if {spec.name for spec in children} != {
            "MN_to_V1Ren_nAChR", "MN_to_V1Ren_GluR",
        } or len({
            (spec.connection_probability, spec.delay_bins_ms)
            for spec in children
        }) != 1:
            raise ValueError(
                "MN-to-Renshaw child paths must share pair topology and "
                f"delay support for population pair {population_pair}"
            )

    for cell_class in CLASSES:
        contract = CLASS_EXECUTION_CONTRACTS[cell_class]
        class_outgoing = {
            spec.name for spec in checked
            if POPULATIONS[spec.source_population].rsplit("_", 2)[0] == cell_class
        }
        class_incoming = {
            spec.name for spec in checked
            if POPULATIONS[spec.target_population].rsplit("_", 2)[0] == cell_class
        }
        if class_outgoing != set(contract.outgoing_pathways.all_names()):
            raise ValueError(f"{cell_class} outgoing pathway-name contract drift")
        if class_incoming != set(contract.incoming_pathways.all_names()):
            raise ValueError(f"{cell_class} incoming pathway-name contract drift")
        for side, phase in SIDE_PHASES:
            population = pop(cell_class, side, phase)
            outgoing = [
                spec.name for spec in checked
                if spec.source_population == population
            ]
            incoming = [
                spec.name for spec in checked
                if spec.target_population == population
            ]
            expected_outgoing = contract.outgoing_pathways.for_phase(phase)
            expected_incoming = contract.incoming_pathways.for_phase(phase)
            if len(outgoing) != len(set(outgoing)) or set(outgoing) != set(
                expected_outgoing
            ):
                raise ValueError(
                    f"{population_name(cell_class, side, phase)} outgoing topology drift"
                )
            if len(incoming) != len(set(incoming)) or set(incoming) != set(
                expected_incoming
            ):
                raise ValueError(
                    f"{population_name(cell_class, side, phase)} incoming topology drift"
                )

    if set(V3_CONNECTIVITY_SUBPHENOTYPE_CONTRACTS) != {"V3_VLat"}:
        raise ValueError("V3 connectivity-subphenotype registry drift")
    vlat_contract = V3_CONNECTIVITY_SUBPHENOTYPE_CONTRACTS["V3_VLat"]
    if (
        vlat_contract.subphenotype != "V3_VLat"
        or vlat_contract.parent_intrinsic_phenotype != "V3_ventral"
        or vlat_contract.mask_symbol != "v3_vlat_connectivity_mask"
        or vlat_contract.incoming_pathways != ("MN_to_V3_VLat_GluR",)
        or vlat_contract.outgoing_pathways != ("V3_VLat_to_ipsilateral_MN",)
        or vlat_contract.source_ids != ("CHOPEK2018_V3_MICROCIRCUIT",)
        or vlat_contract.evidence_grade != "A"
    ):
        raise ValueError("V3-VLat connectivity-subphenotype contract drift")
    for pathway_name in (
        *vlat_contract.incoming_pathways, *vlat_contract.outgoing_pathways,
    ):
        binding = PATHWAY_EVIDENCE[pathway_name]
        if (
            binding.mechanism_evidence != vlat_contract.evidence_grade
            or not set(vlat_contract.source_ids) <= set(binding.source_ids)
        ):
            raise ValueError(f"{pathway_name} V3-VLat evidence binding drift")


V3_CROSS_MOTOR_PATHWAYS = frozenset((
    "V3_ventral_to_contralateral_MN_flexor",
    "V3_ventral_to_contralateral_MN_extensor",
))
V3_VLAT_MICROCIRCUIT_PATHWAYS = frozenset((
    "V3_VLat_to_ipsilateral_MN",
    "MN_to_V3_VLat_GluR",
))
V3_MOTOR_OUTPUT_PATHWAYS = V3_CROSS_MOTOR_PATHWAYS | frozenset((
    "V3_VLat_to_ipsilateral_MN",
))
# Compatibility name retained for the cross-motor assertion in existing
# audits; subtype selection itself is carried by typed Pathway fields.
V3_VENTRAL_MOTOR_PATHWAYS = V3_CROSS_MOTOR_PATHWAYS


def allocate_v3_dorsal_mask(
    cfg: Config,
    structural_seed: int,
    sizes: np.ndarray,
) -> np.ndarray:
    """Freeze the V3 intrinsic/output subtype mask before edge sampling."""
    offsets = np.cumsum(np.r_[0, sizes[:-1]])
    mask = np.zeros(int(np.sum(sizes)), dtype=bool)
    rng = np.random.default_rng(structural_seed + 250527)
    for side, phase in SIDE_PHASES:
        population_id = pop("V3", side, phase)
        ids = offsets[population_id] + np.arange(int(sizes[population_id]))
        labels = allocate_fractional_labels(
            len(ids),
            (1.0 - cfg.v3_dorsal_fraction, cfg.v3_dorsal_fraction),
            rng,
        )
        mask[ids] = labels == 1
    return mask


def allocate_v3_vlat_connectivity_mask(
    cfg: Config,
    structural_seed: int,
    sizes: np.ndarray,
    v3_dorsal_mask: np.ndarray,
) -> np.ndarray:
    """Select an outcome-blind V3-VLat connectivity subset inside ventral V3.

    Chopek 2018 establishes the connectivity phenotype, not its prevalence in
    the modeled four side/phase pools. The configured fraction is therefore a
    declared H-level quota. At least one VLat source/target is retained per
    modeled pool so a named executable subtype cannot disappear at finite N.
    The ventral complement is intentionally left biologically unresolved and
    is never called V3-VMed.
    """
    offsets = np.cumsum(np.r_[0, sizes[:-1]])
    mask = np.zeros(int(np.sum(sizes)), dtype=bool)
    dorsal = np.asarray(v3_dorsal_mask, dtype=bool)
    if dorsal.shape != mask.shape:
        raise ValueError("v3_dorsal_mask shape mismatch")
    rng = np.random.default_rng(structural_seed + 250529)
    for side, phase in SIDE_PHASES:
        population_id = pop("V3", side, phase)
        ids = offsets[population_id] + np.arange(int(sizes[population_id]))
        ventral_ids = ids[~dorsal[ids]]
        if not len(ventral_ids):
            raise ValueError("V3-VLat allocation requires ventral V3 neurons")
        n_selected = max(1, min(
            len(ventral_ids),
            int(math.floor(
                cfg.v3_vlat_fraction_of_ventral * len(ventral_ids) + 0.5
            )),
        ))
        selected = np.sort(rng.choice(
            ventral_ids, size=n_selected, replace=False
        ))
        mask[selected] = True
    if np.any(mask & dorsal):
        raise RuntimeError("V3-VLat connectivity mask escaped ventral V3")
    return mask


def _legacy_bernoulli_connectome_v2_5(
    cfg: Config,
    structural_seed: int,
    sizes: np.ndarray,
) -> Dict[str, object]:
    """Sample sparse cellular edges while preserving expected population gain."""
    rng = np.random.default_rng(structural_seed)
    offsets = np.cumsum(np.r_[0, sizes[:-1]])
    edge_source: List[int] = []
    edge_target: List[int] = []
    edge_weight: List[float] = []
    edge_delay_ms: List[float] = []
    edge_functional_role: List[str] = []
    edge_mt_route: List[str] = []
    edge_recruitment_axis: List[str] = []
    edge_evidence_class: List[str] = []
    edge_evidence_note: List[str] = []
    edge_transmitter_identity: List[str] = []
    edge_topology_group: List[str] = []
    edge_source_subphenotype: List[str] = []
    edge_target_subphenotype: List[str] = []
    edge_pathway: List[int] = []
    pathways = pathway_specs(cfg)
    v3_dorsal_mask = allocate_v3_dorsal_mask(cfg, structural_seed, sizes)
    v3_vlat_connectivity_mask = allocate_v3_vlat_connectivity_mask(
        cfg, structural_seed, sizes, v3_dorsal_mask
    )
    topology_mask_cache: Dict[
        Tuple[str, int, int, float, str, str], np.ndarray
    ] = {}

    for pathway_index, spec in enumerate(pathways):
        source_class = POPULATIONS[
            spec.source_population
        ].rsplit("_", 2)[0]
        source_contract = CLASS_EXECUTION_CONTRACTS[source_class]
        target_transmitter = dict(
            source_contract.target_specific_transmitters
        ).get(spec.name, source_contract.transmitter)
        n_source = int(sizes[spec.source_population])
        n_target = int(sizes[spec.target_population])
        source_ids = offsets[spec.source_population] + np.arange(n_source)
        if spec.source_subphenotype == "ventral":
            source_ids = source_ids[~v3_dorsal_mask[source_ids]]
        elif spec.source_subphenotype == "vlat":
            source_ids = source_ids[v3_vlat_connectivity_mask[source_ids]]
        if not len(source_ids):
            raise ValueError(
                f"{spec.name} has no {spec.source_subphenotype} source neurons"
            )
        n_available_source = len(source_ids)
        target_ids = offsets[spec.target_population] + np.arange(n_target)
        if spec.target_subphenotype == "ventral":
            target_ids = target_ids[~v3_dorsal_mask[target_ids]]
        elif spec.target_subphenotype == "vlat":
            target_ids = target_ids[v3_vlat_connectivity_mask[target_ids]]
        if not len(target_ids):
            raise ValueError(
                f"{spec.name} has no {spec.target_subphenotype} target neurons"
            )
        n_available_target = len(target_ids)
        topology_key = (
            spec.topology_group, spec.source_population,
            spec.target_population, spec.connection_probability,
            spec.source_subphenotype, spec.target_subphenotype,
        )
        cached_mask = (
            topology_mask_cache.get(topology_key)
            if spec.topology_group != "none" else None
        )
        if cached_mask is None:
            mask = (
                rng.random((n_available_target, n_available_source))
                < spec.connection_probability
            )
            if spec.source_population == spec.target_population:
                mask[target_ids[:, None] == source_ids[None, :]] = False
            # A target with zero afferents is an accidental finite-size lesion,
            # not part of the intended uncertainty axis. Ensure one afferent.
            for target_local in np.where(mask.sum(axis=1) == 0)[0]:
                candidates = np.arange(n_available_source)
                if spec.source_population == spec.target_population:
                    candidates = candidates[
                        source_ids[candidates] != target_ids[target_local]
                    ]
                mask[target_local, int(rng.choice(candidates))] = True
            if spec.topology_group != "none":
                topology_mask_cache[topology_key] = mask.copy()
        else:
            mask = cached_mask.copy()
        target_local, source_local = np.where(mask)
        available_sources = (
            n_available_source - 1
            if spec.source_population == spec.target_population
            else n_available_source
        )
        expected_indegree = max(
            spec.connection_probability * available_sources, 1.0
        )
        per_edge_weight = spec.population_weight_pa / expected_indegree
        delays = rng.choice(np.asarray(spec.delay_bins_ms), size=len(target_local))
        edge_source.extend(source_ids[source_local].tolist())
        edge_target.extend(target_ids[target_local].tolist())
        edge_weight.extend([per_edge_weight] * len(target_local))
        edge_delay_ms.extend(delays.astype(float).tolist())
        edge_functional_role.extend([spec.functional_role] * len(target_local))
        edge_mt_route.extend([spec.mt_route] * len(target_local))
        edge_recruitment_axis.extend([spec.recruitment_axis] * len(target_local))
        edge_evidence_class.extend([spec.evidence_class] * len(target_local))
        edge_evidence_note.extend([spec.evidence_note] * len(target_local))
        edge_transmitter_identity.extend([
            target_transmitter
        ] * len(target_local))
        edge_topology_group.extend([
            spec.topology_group
        ] * len(target_local))
        edge_source_subphenotype.extend([
            spec.source_subphenotype
        ] * len(target_local))
        edge_target_subphenotype.extend([
            spec.target_subphenotype
        ] * len(target_local))
        edge_pathway.extend([pathway_index] * len(target_local))

    source = np.asarray(edge_source, dtype=np.int32)
    outgoing: List[np.ndarray] = []
    for neuron in range(int(np.sum(sizes))):
        outgoing.append(np.flatnonzero(source == neuron))
    return {
        "source": source,
        "target": np.asarray(edge_target, dtype=np.int32),
        "weight_pa": np.asarray(edge_weight, dtype=float),
        "delay_ms": np.asarray(edge_delay_ms, dtype=float),
        "functional_role": np.asarray(edge_functional_role, dtype="U24"),
        "mt_route": np.asarray(edge_mt_route, dtype="U16"),
        "recruitment_axis": np.asarray(edge_recruitment_axis, dtype="U16"),
        "evidence_class": np.asarray(edge_evidence_class, dtype="U32"),
        "evidence_note": np.asarray(edge_evidence_note, dtype="U120"),
        "transmitter_identity": np.asarray(
            edge_transmitter_identity, dtype="U32"
        ),
        "topology_group": np.asarray(edge_topology_group, dtype="U32"),
        "source_subphenotype": np.asarray(
            edge_source_subphenotype, dtype="U16"
        ),
        "target_subphenotype": np.asarray(
            edge_target_subphenotype, dtype="U16"
        ),
        "pathway_index": np.asarray(edge_pathway, dtype=np.int16),
        "pathways": pathways,
        "outgoing": outgoing,
        "v3_dorsal_mask": v3_dorsal_mask,
        "v3_vlat_connectivity_mask": v3_vlat_connectivity_mask,
    }


def keyed_structural_rng(
    structural_seed: int,
    label: str,
) -> np.random.Generator:
    """Independent deterministic stream for one structural construction."""
    payload = f"{structural_seed}|{label}".encode()
    local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(local_seed)


def allocate_mirrored_v3_masks(
    cfg: Config,
    structural_seed: int,
    sizes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Allocate homologous V3 intrinsic/VLat masks in all four contexts.

    The finite reduced network represents four copies of the same biological
    dorsoventral composition.  Sharing the relative mask prevents accidental
    left/right or flexor/extensor subtype-count lesions while retaining the
    predeclared V3 fractions and a structural-seed-dependent realization.
    """
    offsets = np.cumsum(np.r_[0, sizes[:-1]])
    dorsal = np.zeros(int(np.sum(sizes)), dtype=bool)
    vlat = np.zeros_like(dorsal)
    n_v3 = int(sizes[pop("V3", "L", "F")])
    local_dorsal = allocate_fractional_labels(
        n_v3,
        (1.0 - cfg.v3_dorsal_fraction, cfg.v3_dorsal_fraction),
        keyed_structural_rng(structural_seed, "V3_dorsal_mirrored"),
    ) == 1
    ventral_local = np.flatnonzero(~local_dorsal)
    if not len(ventral_local):
        raise ValueError("mirrored V3 allocation requires ventral neurons")
    n_vlat = max(1, min(
        len(ventral_local),
        int(math.floor(
            cfg.v3_vlat_fraction_of_ventral * len(ventral_local) + 0.5
        )),
    ))
    local_vlat = np.zeros(n_v3, dtype=bool)
    local_vlat[np.sort(keyed_structural_rng(
        structural_seed, "V3_VLat_mirrored"
    ).choice(ventral_local, size=n_vlat, replace=False))] = True
    for side, phase in SIDE_PHASES:
        population_id = pop("V3", side, phase)
        ids = offsets[population_id] + np.arange(n_v3)
        dorsal[ids] = local_dorsal
        vlat[ids] = local_vlat
    if np.any(vlat & dorsal):
        raise RuntimeError("V3-VLat mask escaped ventral V3")
    return dorsal, vlat


def fixed_indegree_mask(
    n_target: int,
    n_source: int,
    probability: float,
    same_population: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct a sparse mask with identical target indegree."""
    available = n_source - 1 if same_population else n_source
    if available <= 0:
        raise ValueError("pathway has no available source")
    indegree = max(1, min(
        available,
        int(math.floor(probability * available + 0.5)),
    ))
    mask = np.zeros((n_target, n_source), dtype=bool)
    if same_population:
        if n_target != n_source:
            raise ValueError("recurrent pathway population sizes drifted")
        relative_sources = np.sort(rng.choice(
            np.arange(1, n_source), size=indegree, replace=False
        ))
        for target in range(n_target):
            mask[target, (target + relative_sources) % n_source] = True
    else:
        source_order = rng.permutation(n_source)
        target_order = rng.permutation(n_target)
        phase = int(rng.integers(0, n_source))
        for rank, target in enumerate(target_order):
            positions = (
                phase + rank * indegree + np.arange(indegree)
            ) % n_source
            mask[target, source_order[positions]] = True
    if not np.all(mask.sum(axis=1) == indegree):
        raise RuntimeError("fixed-indegree construction failed")
    return mask


def balanced_delay_assignment(
    count: int,
    bins: Tuple[float, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Cycle deterministically over a seeded permutation of physical bins."""
    order = rng.permutation(len(bins))
    phase = int(rng.integers(0, len(bins)))
    indices = order[(phase + np.arange(count)) % len(bins)]
    return np.asarray(bins, dtype=float)[indices]


def build_connectome(
    cfg: Config,
    structural_seed: int,
    sizes: np.ndarray,
) -> Dict[str, object]:
    """Build the v2.6 fixed-indegree homologous mirrored connectome.

    Named pathways, signs, delays, target subphenotypes and total declared
    population gains are unchanged.  The numerical realization removes
    unintended finite-N zero-afferent lesions and sampling imbalance: every
    target has the same pathway indegree, homologous copies share one relative
    sparse template, and each target receives exactly the declared total gain.
    """
    offsets = np.cumsum(np.r_[0, sizes[:-1]])
    pathways = pathway_specs(cfg)
    v3_dorsal_mask, v3_vlat_connectivity_mask = allocate_mirrored_v3_masks(
        cfg, structural_seed, sizes
    )
    edge_source: List[int] = []
    edge_target: List[int] = []
    edge_weight: List[float] = []
    edge_delay_ms: List[float] = []
    edge_functional_role: List[str] = []
    edge_mt_route: List[str] = []
    edge_recruitment_axis: List[str] = []
    edge_evidence_class: List[str] = []
    edge_evidence_note: List[str] = []
    edge_transmitter_identity: List[str] = []
    edge_topology_group: List[str] = []
    edge_source_subphenotype: List[str] = []
    edge_target_subphenotype: List[str] = []
    edge_pathway: List[int] = []
    topology_cache: Dict[
        Tuple[str, int, int, float, bool], np.ndarray
    ] = {}

    for pathway_index, spec in enumerate(pathways):
        source_class = POPULATIONS[
            spec.source_population
        ].rsplit("_", 2)[0]
        source_contract = CLASS_EXECUTION_CONTRACTS[source_class]
        target_transmitter = dict(
            source_contract.target_specific_transmitters
        ).get(spec.name, source_contract.transmitter)
        source_ids = offsets[spec.source_population] + np.arange(
            int(sizes[spec.source_population])
        )
        target_ids = offsets[spec.target_population] + np.arange(
            int(sizes[spec.target_population])
        )
        if spec.source_subphenotype == "ventral":
            source_ids = source_ids[~v3_dorsal_mask[source_ids]]
        elif spec.source_subphenotype == "vlat":
            source_ids = source_ids[v3_vlat_connectivity_mask[source_ids]]
        if spec.target_subphenotype == "ventral":
            target_ids = target_ids[~v3_dorsal_mask[target_ids]]
        elif spec.target_subphenotype == "vlat":
            target_ids = target_ids[v3_vlat_connectivity_mask[target_ids]]
        if not len(source_ids) or not len(target_ids):
            raise ValueError(f"{spec.name} lost a required subphenotype")

        same_population = spec.source_population == spec.target_population
        template_name = (
            spec.topology_group
            if spec.topology_group != "none" else spec.name
        )
        cache_key = (
            template_name,
            len(target_ids),
            len(source_ids),
            float(spec.connection_probability),
            same_population,
        )
        mask = topology_cache.get(cache_key)
        if mask is None:
            mask = fixed_indegree_mask(
                len(target_ids),
                len(source_ids),
                float(spec.connection_probability),
                same_population,
                keyed_structural_rng(
                    structural_seed, f"topology|{template_name}"
                ),
            )
            topology_cache[cache_key] = mask.copy()
        else:
            mask = mask.copy()

        target_local, source_local = np.where(mask)
        indegree = mask.sum(axis=1)
        if not np.all(indegree == indegree[0]):
            raise RuntimeError("pathway target indegree is not balanced")
        per_edge_weight = (
            float(spec.population_weight_pa) / int(indegree[0])
        )
        delays = balanced_delay_assignment(
            len(target_local),
            tuple(spec.delay_bins_ms),
            keyed_structural_rng(
                structural_seed,
                f"delay|{spec.name}|{spec.source_subphenotype}|"
                f"{spec.target_subphenotype}",
            ),
        )
        edge_source.extend(source_ids[source_local].tolist())
        edge_target.extend(target_ids[target_local].tolist())
        edge_weight.extend([per_edge_weight] * len(target_local))
        edge_delay_ms.extend(delays.tolist())
        edge_functional_role.extend([spec.functional_role] * len(target_local))
        edge_mt_route.extend([spec.mt_route] * len(target_local))
        edge_recruitment_axis.extend([spec.recruitment_axis] * len(target_local))
        edge_evidence_class.extend([spec.evidence_class] * len(target_local))
        edge_evidence_note.extend([spec.evidence_note] * len(target_local))
        edge_transmitter_identity.extend([target_transmitter] * len(target_local))
        edge_topology_group.extend([spec.topology_group] * len(target_local))
        edge_source_subphenotype.extend([
            spec.source_subphenotype
        ] * len(target_local))
        edge_target_subphenotype.extend([
            spec.target_subphenotype
        ] * len(target_local))
        edge_pathway.extend([pathway_index] * len(target_local))

    source = np.asarray(edge_source, dtype=np.int32)
    outgoing = [
        np.flatnonzero(source == neuron)
        for neuron in range(int(np.sum(sizes)))
    ]
    return {
        "source": source,
        "target": np.asarray(edge_target, dtype=np.int32),
        "weight_pa": np.asarray(edge_weight, dtype=float),
        "delay_ms": np.asarray(edge_delay_ms, dtype=float),
        "functional_role": np.asarray(edge_functional_role, dtype="U24"),
        "mt_route": np.asarray(edge_mt_route, dtype="U16"),
        "recruitment_axis": np.asarray(edge_recruitment_axis, dtype="U16"),
        "evidence_class": np.asarray(edge_evidence_class, dtype="U32"),
        "evidence_note": np.asarray(edge_evidence_note, dtype="U120"),
        "transmitter_identity": np.asarray(
            edge_transmitter_identity, dtype="U32"
        ),
        "topology_group": np.asarray(edge_topology_group, dtype="U32"),
        "source_subphenotype": np.asarray(
            edge_source_subphenotype, dtype="U16"
        ),
        "target_subphenotype": np.asarray(
            edge_target_subphenotype, dtype="U16"
        ),
        "pathway_index": np.asarray(edge_pathway, dtype=np.int16),
        "pathways": pathways,
        "outgoing": outgoing,
        "v3_dorsal_mask": v3_dorsal_mask,
        "v3_vlat_connectivity_mask": v3_vlat_connectivity_mask,
    }


def simulate(
    cfg: Config,
    seed: int,
    protocol: str,
    mt_mode: str,
    structural_seed: int | None = None,
    static_scale: float = 1.0,
    fast_mode: str = "dynamic",
    ablated_pathways: Sequence[str] = (),
    disabled_intrinsic_mechanisms: Sequence[str] = (),
    external_kca_event_steps: np.ndarray | None = None,
    external_kca_event_neurons: np.ndarray | None = None,
    external_kca_event_times_s: np.ndarray | None = None,
    external_mt_event_steps: np.ndarray | None = None,
    external_mt_event_edges: np.ndarray | None = None,
    external_mt_event_times_s: np.ndarray | None = None,
    fast_activation_scale: float = 1.0,
    speed_level: str = "medium",
    load_context: str = "normal",
    load_side: str = "L",
    pulse_direction: str = "none",
    pulse_cycle_fraction_override: float | None = None,
    pulse_target_side: str = "R",
    pulse_target_phase: str = "F",
    ablated_populations: Sequence[str] = (),
    impaired_mt_routes: Sequence[str] = (),
    challenged_routes: Sequence[str] = MT_ROUTES,
) -> Dict[str, np.ndarray]:
    if protocol not in {"steady_state", "pulse", "long", "noise_burst", "speed_step", "phase_kick", "pf_deletion"}:
        raise ValueError(protocol)
    validate_config(cfg, protocol=protocol)
    validate_pathway_class_coverage(cfg)
    if mt_mode not in MT_MODES:
        raise ValueError(mt_mode)
    if fast_mode not in {"off", "dynamic", "static_mean", "yoked"}:
        raise ValueError(fast_mode)
    if speed_level not in SPEED_LEVELS:
        raise ValueError(f"Unknown speed level: {speed_level}")
    if load_context not in LOAD_CONTEXTS:
        raise ValueError(f"Unknown load context: {load_context}")
    if load_side not in SIDES or pulse_target_side not in SIDES:
        raise ValueError("load_side and pulse_target_side must be L or R")
    if pulse_target_phase not in PHASES:
        raise ValueError("pulse_target_phase must be F or E")
    if pulse_direction not in PULSE_DIRECTIONS:
        raise ValueError(f"Unknown pulse direction: {pulse_direction}")
    if pulse_cycle_fraction_override is not None and not 0.0 < pulse_cycle_fraction_override < 1.0:
        raise ValueError("pulse_cycle_fraction_override must be in (0, 1)")
    unknown_population_ablations = set(ablated_populations) - (set(CLASSES) | set(AFFERENT_ABLATIONS))
    if unknown_population_ablations:
        raise ValueError(f"Unknown population ablations: {sorted(unknown_population_ablations)}")
    if len(disabled_intrinsic_mechanisms) != len(
        set(disabled_intrinsic_mechanisms)
    ):
        raise ValueError("disabled_intrinsic_mechanisms may not contain duplicates")
    unknown_disabled_mechanisms = set(
        disabled_intrinsic_mechanisms
    ) - DISABLABLE_INTRINSIC_TERM_IDS
    if unknown_disabled_mechanisms:
        raise ValueError(
            "Unknown or non-ablatable intrinsic mechanisms: "
            f"{sorted(unknown_disabled_mechanisms)}"
        )
    disabled_intrinsic_mechanism_set = set(disabled_intrinsic_mechanisms)
    unknown_impaired_routes = set(impaired_mt_routes) - set(MT_ROUTES)
    unknown_challenged_routes = set(challenged_routes) - set(MT_ROUTES)
    if unknown_impaired_routes or unknown_challenged_routes:
        raise ValueError(
            f"Unknown MT routes: {sorted(unknown_impaired_routes | unknown_challenged_routes)}"
        )
    expected_long_duration = cfg.long_n_epochs * cfg.long_epoch_duration_s
    if protocol == "long" and not math.isclose(
        cfg.duration_s, expected_long_duration, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "Long protocol duration must equal "
            "long_n_epochs * long_epoch_duration_s"
        )

    structural_seed = int(100000 + seed if structural_seed is None else structural_seed)
    noise_rng = np.random.default_rng(seed)
    structure_rng = np.random.default_rng(structural_seed + 991)
    sizes, neuron_pop, neuron_local, mt_eligible_pop = population_metadata(cfg)
    n_neuron = len(neuron_pop)
    n_pop = len(POPULATIONS)
    is_rg = np.asarray([POPULATIONS[p].startswith("RG_") for p in neuron_pop])
    is_mn = np.asarray([POPULATIONS[p].startswith("MN_") for p in neuron_pop])
    neuron_class = np.asarray([POPULATIONS[p].rsplit("_", 2)[0] for p in neuron_pop])
    class_index = {cell_class: index for index, cell_class in enumerate(CLASSES)}
    neuron_class_index = np.asarray(
        [class_index[cell_class] for cell_class in neuron_class], dtype=np.int8
    )
    class_neuron_count = np.bincount(
        neuron_class_index, minlength=len(CLASSES)
    ).astype(float)
    class_neuron_start = np.r_[
        0, np.cumsum(class_neuron_count[:-1], dtype=int)
    ].astype(int)
    is_pf = neuron_class == "PF"
    is_v2a = neuron_class == "V2a"
    is_v3 = neuron_class == "V3"
    is_renshaw = neuron_class == "V1Ren"
    # A dedicated stream keeps the literature-derived RG subtype allocation
    # reproducible without changing any pre-existing connectome/noise draws.
    rg_subtype_rng = np.random.default_rng(structural_seed + 250525)
    rg_pic_positive_mask = allocate_class_population_mask(
        neuron_pop, "RG", cfg.rg_pic_positive_fraction, rg_subtype_rng
    )
    rg_m_positive_mask = allocate_class_population_mask(
        neuron_pop, "RG", cfg.rg_m_positive_fraction, rg_subtype_rng
    )
    rg_kca_positive_mask = allocate_class_population_mask(
        neuron_pop, "RG", cfg.rg_kca_positive_fraction, rg_subtype_rng
    )
    rg_h_positive_mask = allocate_class_population_mask(
        neuron_pop, "RG", cfg.rg_h_positive_fraction, rg_subtype_rng
    )
    rg_t_positive_mask = allocate_class_population_mask(
        neuron_pop, "RG", cfg.rg_t_positive_fraction, rg_subtype_rng
    )
    rg_a_positive_mask = allocate_class_population_mask(
        neuron_pop, "RG", cfg.rg_a_positive_fraction, rg_subtype_rng
    )
    ablated_population_set = set(ablated_populations)
    ablated_neuron = np.isin(
        neuron_class, list(ablated_population_set - set(AFFERENT_ABLATIONS))
    )
    ablated_population_mask = np.asarray([
        POPULATIONS[p].rsplit("_", 2)[0] in ablated_population_set
        for p in range(n_pop)
    ])
    ia_ablated = "Ia" in ablated_population_set or "groupI" in ablated_population_set
    ib_ablated = "Ib" in ablated_population_set or "groupI" in ablated_population_set
    connectome = build_connectome(cfg, structural_seed, sizes)
    v3_dorsal_mask = np.asarray(
        connectome["v3_dorsal_mask"], dtype=bool
    )
    v3_vlat_connectivity_mask = np.asarray(
        connectome["v3_vlat_connectivity_mask"], dtype=bool
    )
    if np.any(v3_vlat_connectivity_mask & v3_dorsal_mask) or np.any(
        v3_vlat_connectivity_mask & ~is_v3
    ):
        raise RuntimeError("V3-VLat mask must be a subset of ventral V3")
    known_pathways = {spec.name for spec in connectome["pathways"]}
    unknown_ablations = set(ablated_pathways) - known_pathways
    if unknown_ablations:
        raise ValueError(f"Unknown pathway ablations: {sorted(unknown_ablations)}")
    ablated_pathway_set = set(ablated_pathways)
    ablated_indices = {
        index for index, spec in enumerate(connectome["pathways"])
        if spec.name in ablated_pathway_set
    }
    ablated_edge_mask = np.isin(connectome["pathway_index"], list(ablated_indices))
    ablated_edge_mask |= ablated_neuron[connectome["source"]]
    ablated_edge_mask |= ablated_neuron[connectome["target"]]
    route_index = {name: index for index, name in enumerate(MT_ROUTES)}
    edge_route_index = np.asarray([
        route_index.get(name, -1) for name in connectome["mt_route"]
    ], dtype=np.int16)
    expected_n_step = int(round(cfg.duration_s * 1000.0 / cfg.dt_ms))
    simulation_end_s = expected_n_step * cfg.dt_ms / 1000.0
    if (
        external_kca_event_times_s is not None
        and external_kca_event_steps is not None
    ):
        raise ValueError(
            "provide physical KCa event times or legacy event steps, not both"
        )
    if fast_mode != "yoked" and any(
        value is not None
        for value in (
            external_kca_event_times_s,
            external_kca_event_steps,
            external_kca_event_neurons,
        )
    ):
        raise ValueError("external KCa events are valid only in yoked fast mode")
    kca_time_api = external_kca_event_times_s is not None
    if fast_mode == "yoked":
        if external_kca_event_neurons is None or (
            external_kca_event_steps is None
            and external_kca_event_times_s is None
        ):
            raise ValueError(
                "yoked fast mode requires external KCa event times (preferred) "
                "or legacy steps plus neuron ids"
            )
        kca_event_neurons = np.asarray(external_kca_event_neurons, dtype=np.int64)
        if np.any(kca_event_neurons < 0) or np.any(kca_event_neurons >= n_neuron):
            raise ValueError("external KCa event neuron outside model")
        if np.any(~rg_kca_positive_mask[kca_event_neurons]):
            raise ValueError(
                "external KCa events must target the frozen KCa-positive RG subset"
            )
        if kca_time_api:
            kca_event_times_s = np.asarray(
                external_kca_event_times_s, dtype=float
            )
            if kca_event_times_s.shape != kca_event_neurons.shape:
                raise ValueError(
                    "external KCa event-time and neuron arrays must have identical shape"
                )
            if np.any(~np.isfinite(kca_event_times_s)) or np.any(
                (kca_event_times_s < 0.0)
                | (kca_event_times_s > simulation_end_s)
            ):
                raise ValueError("external KCa event time outside simulation")
            order = np.argsort(kca_event_times_s, kind="stable")
            kca_event_times_s = kca_event_times_s[order]
            kca_event_steps = np.asarray([], dtype=np.int64)
        else:
            kca_event_steps = np.asarray(
                external_kca_event_steps, dtype=np.int64
            )
            if kca_event_steps.shape != kca_event_neurons.shape:
                raise ValueError(
                    "external KCa event-step and neuron arrays must have identical shape"
                )
            if np.any(kca_event_steps < 0) or np.any(
                kca_event_steps >= expected_n_step
            ):
                raise ValueError("external KCa event step outside simulation")
            order = np.argsort(kca_event_steps, kind="stable")
            kca_event_steps = kca_event_steps[order]
            kca_event_times_s = np.asarray([], dtype=float)
        kca_event_neurons = kca_event_neurons[order]
    else:
        kca_event_steps = np.asarray([], dtype=np.int64)
        kca_event_times_s = np.asarray([], dtype=float)
        kca_event_neurons = np.asarray([], dtype=np.int64)
    kca_event_cursor = 0

    # v2.5 rule: no generic "relay" fallback. Every neuron reads the equation
    # parameters of its declared biological/functional class.
    equation_records = [CELL_CLASS_EQUATIONS[cell_class] for cell_class in neuron_class]
    capacitance = np.asarray([record.capacitance_pf for record in equation_records])
    leak = np.asarray([record.leak_ns for record in equation_records])
    adaptation_a = np.asarray([record.adaptation_a_ns for record in equation_records])
    adaptation_tau = np.asarray([record.adaptation_tau_ms for record in equation_records])
    adaptation_b = np.asarray([record.adaptation_b_pa for record in equation_records])
    speed_index = SPEED_LEVELS.index(speed_level)
    nap_conductance = np.where(
        rg_pic_positive_mask, cfg.rg_nap_conductance_ns, 0.0
    )
    base_tonic_drive = np.asarray([
        record.tonic_drive_pa for record in equation_records
    ])
    tonic_drive = base_tonic_drive.copy()
    tonic_drive[is_rg] += cfg.descending_rg_drive_offsets_pa[speed_index]

    # Positive bounded heterogeneity can vary magnitude but can never reverse
    # the typed sign of a direct drive or transmitter current.
    drive_scale = np.clip(
        structure_rng.normal(1.0, cfg.drive_heterogeneity_fraction, n_neuron),
        0.5, 1.5,
    )
    syn_scale = np.clip(
        structure_rng.normal(1.0, cfg.synaptic_heterogeneity_fraction, n_neuron),
        0.5, 1.5,
    )
    nap_scale = np.clip(structure_rng.normal(1.0, 0.05, n_neuron), 0.75, 1.25)
    v = noise_rng.normal(cfg.leak_reversal_mv + 3.0, 2.5, n_neuron)
    mn_dendrite_v = v.copy()
    mn_nap_pic_inactivation = sigmoid(
        -(mn_dendrite_v - cfg.mn_nap_pic_inactivation_half_mv)
        / cfg.mn_nap_pic_inactivation_slope_mv
    )
    mn_ltype_ca_pic_activation = sigmoid(
        (mn_dendrite_v - cfg.mn_ltype_ca_pic_activation_half_mv)
        / cfg.mn_ltype_ca_pic_activation_slope_mv
    )
    w = np.zeros(n_neuron)
    h = sigmoid(-(v - cfg.nap_inactivation_half_mv) / cfg.nap_inactivation_slope_mv)
    rg_ltype_ca_pic_activation = sigmoid(
        (v - cfg.rg_ltype_ca_pic_activation_half_mv)
        / cfg.rg_ltype_ca_pic_activation_slope_mv
    )
    rg_h_gate = sigmoid(-(v - cfg.rg_h_half_mv) / cfg.rg_h_slope_mv)
    rg_t_inactivation = sigmoid(
        -(v - cfg.rg_t_inactivation_half_mv)
        / cfg.rg_t_inactivation_slope_mv
    )
    rg_a_inactivation = sigmoid(
        -(v - cfg.rg_a_inactivation_half_mv)
        / cfg.rg_a_inactivation_slope_mv
    )
    m_gate = sigmoid((v - cfg.m_activation_half_mv) / cfg.m_activation_slope_mv)
    calcium = np.zeros(n_neuron)
    mn_calcium = np.zeros(n_neuron)
    pf_slow_gate = np.zeros(n_neuron)

    # Directly measured V2a firing phenotypes are assigned within each V2a
    # population with the same frozen structural seed. 0=tonic, 1=phasic,
    # 2=delayed. Delayed cells receive a phenomenological onset-delay state
    # with no asserted channel identity; phasic cells receive stronger
    # spike-triggered adaptation.
    v2a_variant = np.full(n_neuron, -1, dtype=np.int8)
    v2a_ids_by_context = tuple(
        np.flatnonzero(neuron_pop == pop("V2a", side, phase))
        for side, phase in SIDE_PHASES
    )
    v2a_labels_by_context = allocate_grouped_fractional_labels(
        tuple(len(ids) for ids in v2a_ids_by_context),
        cfg.v2a_variant_fractions,
        structure_rng,
    )
    for v2a_ids, labels in zip(v2a_ids_by_context, v2a_labels_by_context):
        v2a_variant[v2a_ids] = labels
    adaptation_b[v2a_variant == 1] *= cfg.v2a_phasic_adaptation_multiplier
    v2a_delay_relief = sigmoid(
        (v - cfg.v2a_delayed_activation_half_mv)
        / cfg.v2a_delayed_activation_slope_mv
    )
    v2a_h_positive_mask = np.zeros(n_neuron, dtype=bool)
    # Zhong et al. report phenotype-level prevalences across their sample, not
    # one quota per side/phase.  Round once across all four local contexts so
    # the default two-cell phasic pools do not turn 26/29 into 1/2 each.
    for variant_id, positive_fraction in enumerate(
        cfg.v2a_h_positive_fractions
    ):
        variant_ids = np.flatnonzero(is_v2a & (v2a_variant == variant_id))
        if not len(variant_ids):
            continue
        if positive_fraction == 1.0:
            selected = variant_ids
        else:
            n_selected = max(1, min(
                len(variant_ids) - 1,
                int(math.floor(positive_fraction * len(variant_ids) + 0.5)),
            ))
            selected = np.sort(structure_rng.choice(
                variant_ids, size=n_selected, replace=False
            ))
        v2a_h_positive_mask[selected] = True
    v2a_h_gate = sigmoid(-(v - cfg.v2a_h_half_mv) / cfg.v2a_h_slope_mv)

    # Electrical-coupling measurements are pair incidences, so they are
    # realized as symmetric sparse pair graphs rather than cell-prevalence
    # masks or all-to-mean currents. All edges are local to one side/phase;
    # V2a tonic and phasic incidences are sampled separately and mixed/delayed
    # pairs are absent from the represented evidence-supported graph.
    v2a_gap_rng = np.random.default_rng(structural_seed + 250526)
    rg_gap_groups: List[np.ndarray] = []
    v2a_tonic_gap_groups: List[np.ndarray] = []
    v2a_phasic_gap_groups: List[np.ndarray] = []
    for side, phase in SIDE_PHASES:
        rg_gap_groups.append(np.flatnonzero(
            neuron_pop == pop("RG", side, phase)
        ))
        population_mask = neuron_pop == pop("V2a", side, phase)
        v2a_tonic_gap_groups.append(np.flatnonzero(
            population_mask & (v2a_variant == 0)
        ))
        v2a_phasic_gap_groups.append(np.flatnonzero(
            population_mask & (v2a_variant == 1)
        ))
    rg_gap_source, rg_gap_target, rg_gap_candidate_pair_count = (
        sample_grouped_symmetric_pair_edges(
            rg_gap_groups, cfg.rg_gap_pair_probability, rg_subtype_rng
        )
    )
    (
        v2a_tonic_gap_source,
        v2a_tonic_gap_target,
        v2a_tonic_gap_candidate_pair_count,
    ) = sample_grouped_symmetric_pair_edges(
        v2a_tonic_gap_groups,
        cfg.v2a_tonic_gap_pair_probability,
        v2a_gap_rng,
    )
    (
        v2a_phasic_gap_source,
        v2a_phasic_gap_target,
        v2a_phasic_gap_candidate_pair_count,
    ) = sample_grouped_symmetric_pair_edges(
        v2a_phasic_gap_groups,
        cfg.v2a_phasic_gap_pair_probability,
        v2a_gap_rng,
    )
    # Own-voltage derivatives of the symmetric gap currents.  These are used
    # only by the local numerical Jacobian; the physical gap-current equations
    # and sampled pair graphs are unchanged.
    rg_gap_degree = np.bincount(
        np.r_[rg_gap_source, rg_gap_target], minlength=n_neuron
    ).astype(float)
    v2a_tonic_gap_degree = np.bincount(
        np.r_[v2a_tonic_gap_source, v2a_tonic_gap_target],
        minlength=n_neuron,
    ).astype(float)
    v2a_phasic_gap_degree = np.bincount(
        np.r_[v2a_phasic_gap_source, v2a_phasic_gap_target],
        minlength=n_neuron,
    ).astype(float)

    # V3 subtype identity is frozen before connectome sampling so only the
    # ventral subset can source the literature-supported motor-pool paths.
    # Dorsal cells retain distinct Ih/T/adaptation but receive no invented
    # motor projection.
    adaptation_b[v3_dorsal_mask] *= cfg.v3_dorsal_adaptation_multiplier
    v3_h_gate = sigmoid(-(v - cfg.v3_h_half_mv) / cfg.v3_h_slope_mv)
    v3_t_inactivation = sigmoid(
        -(v - cfg.v3_t_inactivation_half_mv)
        / cfg.v3_t_inactivation_slope_mv
    )

    renshaw_h_gate = sigmoid(
        -(v - cfg.renshaw_h_half_mv) / cfg.renshaw_h_slope_mv
    )
    renshaw_calcium = np.zeros(n_neuron)
    renshaw_nachr_conductance = np.zeros(n_neuron)
    renshaw_glutamate_conductance = np.zeros(n_neuron)
    eta = np.zeros(n_neuron)
    eta_pop = np.zeros(n_pop)
    refractory = np.zeros(n_neuron)
    syn_exc_conductance = np.zeros(n_neuron)
    syn_inh_conductance = np.zeros(n_neuron)
    rate = np.zeros(n_pop)

    dt = cfg.dt_ms
    ring_length = int(math.ceil(float(np.max(connectome["delay_ms"])) / dt)) + 3
    scheduled_exc_conductance = np.zeros((ring_length, n_neuron), dtype=float)
    scheduled_inh_conductance = np.zeros((ring_length, n_neuron), dtype=float)
    scheduled_nachr_conductance = np.zeros((ring_length, n_neuron), dtype=float)
    scheduled_renshaw_glutamate_conductance = np.zeros(
        (ring_length, n_neuron), dtype=float
    )
    ring_pointer = 0
    central_delay_reconstruction_max_abs_error_ms = 0.0
    central_event_split_mass_max_abs_error = 0.0
    renshaw_nachr_pathways = {
        index for index, spec in enumerate(connectome["pathways"])
        if spec.name == "MN_to_V1Ren_nAChR"
    }
    renshaw_nachr_edge_mask = np.isin(
        connectome["pathway_index"], list(renshaw_nachr_pathways)
    )
    renshaw_glutamate_pathways = {
        index for index, spec in enumerate(connectome["pathways"])
        if spec.name == "MN_to_V1Ren_GluR"
    }
    renshaw_glutamate_edge_mask = np.isin(
        connectome["pathway_index"], list(renshaw_glutamate_pathways)
    )
    v3_motor_pathways = {
        index for index, spec in enumerate(connectome["pathways"])
        if spec.name in V3_MOTOR_OUTPUT_PATHWAYS
    }
    v3_motor_edge_mask = np.isin(
        connectome["pathway_index"], list(v3_motor_pathways)
    )
    v3_cross_motor_pathways = {
        index for index, spec in enumerate(connectome["pathways"])
        if spec.name in V3_CROSS_MOTOR_PATHWAYS
    }
    v3_cross_motor_edge_mask = np.isin(
        connectome["pathway_index"], list(v3_cross_motor_pathways)
    )
    v3_vlat_output_pathways = {
        index for index, spec in enumerate(connectome["pathways"])
        if spec.name == "V3_VLat_to_ipsilateral_MN"
    }
    v3_vlat_output_edge_mask = np.isin(
        connectome["pathway_index"], list(v3_vlat_output_pathways)
    )
    mn_v3_vlat_input_pathways = {
        index for index, spec in enumerate(connectome["pathways"])
        if spec.name == "MN_to_V3_VLat_GluR"
    }
    mn_v3_vlat_input_edge_mask = np.isin(
        connectome["pathway_index"], list(mn_v3_vlat_input_pathways)
    )
    v3_microcircuit_pathway_names = np.asarray([
        "V3_VLat_to_ipsilateral_MN", "MN_to_V3_VLat_GluR",
    ])
    v3_microcircuit_edge_masks = np.asarray([
        v3_vlat_output_edge_mask, mn_v3_vlat_input_edge_mask,
    ])
    if np.any(
        v3_dorsal_mask[connectome["source"][v3_motor_edge_mask]]
    ):
        raise RuntimeError("dorsal V3 neuron received a ventral motor edge")
    if np.any(
        ~v3_vlat_connectivity_mask[
            connectome["source"][v3_vlat_output_edge_mask]
        ]
    ):
        raise RuntimeError("non-VLat V3 neuron received a V3-VLat output edge")
    if np.any(
        ~v3_vlat_connectivity_mask[
            connectome["target"][mn_v3_vlat_input_edge_mask]
        ]
    ):
        raise RuntimeError("MN-to-V3-VLat edge targeted a non-VLat neuron")
    if np.any(
        connectome["transmitter_identity"][mn_v3_vlat_input_edge_mask]
        != "glutamate"
    ):
        raise RuntimeError("MN-to-V3-VLat transmission must be glutamatergic")
    for edge_id in np.flatnonzero(
        v3_cross_motor_edge_mask
        | v3_vlat_output_edge_mask
        | mn_v3_vlat_input_edge_mask
    ):
        source_name = POPULATIONS[
            int(neuron_pop[int(connectome["source"][edge_id])])
        ]
        target_name = POPULATIONS[
            int(neuron_pop[int(connectome["target"][edge_id])])
        ]
        source_class, source_side, source_phase = source_name.rsplit("_", 2)
        target_class, target_side, target_phase = target_name.rsplit("_", 2)
        if v3_cross_motor_edge_mask[edge_id]:
            topology_ok = (
                source_class == "V3" and target_class == "MN"
                and source_side != target_side
                and source_phase == target_phase
            )
        elif v3_vlat_output_edge_mask[edge_id]:
            topology_ok = (
                source_class == "V3" and target_class == "MN"
                and source_side == target_side
                and source_phase == target_phase
            )
        else:
            topology_ok = (
                source_class == "MN" and target_class == "V3"
                and source_side == target_side
                and source_phase == target_phase
            )
        if not topology_ok:
            raise RuntimeError("V3 motor-microcircuit side/phase topology drift")
    edge_rrp_available = np.ones(len(connectome["source"]), dtype=float)
    edge_slow_replenishment_resource = np.ones(
        len(connectome["source"]), dtype=float
    )
    edge_mt_route = connectome["mt_route"]
    source_population = neuron_pop[connectome["source"]]
    edge_side_index = np.asarray([
        SIDES.index(POPULATIONS[p].rsplit("_", 2)[1])
        for p in source_population
    ])
    resource_edge_mask = edge_route_index >= 0
    resource_edge_ids = np.flatnonzero(resource_edge_mask)
    if (
        external_mt_event_times_s is not None
        and external_mt_event_steps is not None
    ):
        raise ValueError(
            "provide physical MT event times or legacy event steps, not both"
        )
    if mt_mode != "time_yoked" and any(
        value is not None
        for value in (
            external_mt_event_times_s,
            external_mt_event_steps,
            external_mt_event_edges,
        )
    ):
        raise ValueError(
            "external MT events are valid only in time_yoked MT mode"
        )
    mt_time_api = external_mt_event_times_s is not None
    if mt_mode == "time_yoked":
        if external_mt_event_edges is None or (
            external_mt_event_steps is None
            and external_mt_event_times_s is None
        ):
            raise ValueError(
                "time_yoked MT mode requires external terminal event times "
                "(preferred) or legacy steps plus edge ids"
            )
        mt_event_edges = np.asarray(external_mt_event_edges, dtype=np.int64)
        if np.any(mt_event_edges < 0) or np.any(
            mt_event_edges >= len(edge_rrp_available)
        ):
            raise ValueError("external MT event edge outside connectome")
        if np.any(~resource_edge_mask[mt_event_edges]) or np.any(
            ablated_edge_mask[mt_event_edges]
        ):
            raise ValueError("external MT events must target eligible, intact terminals")
        if mt_time_api:
            mt_event_times_s = np.asarray(
                external_mt_event_times_s, dtype=float
            )
            if mt_event_times_s.shape != mt_event_edges.shape:
                raise ValueError(
                    "external MT event-time and edge arrays must have identical shape"
                )
            if np.any(~np.isfinite(mt_event_times_s)) or np.any(
                (mt_event_times_s < 0.0)
                | (mt_event_times_s > simulation_end_s)
            ):
                raise ValueError("external MT event time outside simulation")
            order = np.argsort(mt_event_times_s, kind="stable")
            mt_event_times_s = mt_event_times_s[order]
            mt_event_steps = np.asarray([], dtype=np.int64)
        else:
            mt_event_steps = np.asarray(
                external_mt_event_steps, dtype=np.int64
            )
            if mt_event_steps.shape != mt_event_edges.shape:
                raise ValueError(
                    "external MT event-step and edge arrays must have identical shape"
                )
            if np.any(mt_event_steps < 0) or np.any(
                mt_event_steps >= expected_n_step
            ):
                raise ValueError("external MT event step outside simulation")
            order = np.argsort(mt_event_steps, kind="stable")
            mt_event_steps = mt_event_steps[order]
            mt_event_times_s = np.asarray([], dtype=float)
        mt_event_edges = mt_event_edges[order]
    else:
        mt_event_steps = np.asarray([], dtype=np.int64)
        mt_event_times_s = np.asarray([], dtype=float)
        mt_event_edges = np.asarray([], dtype=np.int64)
    mt_event_cursor = 0
    mt_edge_activity = np.zeros(len(edge_rrp_available))
    mt_edge_tracks = np.zeros(len(edge_rrp_available))
    mt_edge_tracks[resource_edge_mask] = cfg.mt_initial_track_fraction
    if mt_mode == "off":
        mt_edge_tracks[resource_edge_mask] = 0.0
    elif mt_mode == "static_matched":
        mt_edge_tracks[resource_edge_mask] = np.asarray(
            cfg.mt_static_route_supports
        )[edge_route_index[resource_edge_mask]] * static_scale
    mt_edge_tracks = np.clip(mt_edge_tracks, 0.0, cfg.mt_track_max)
    depletion_fraction = np.full(
        len(edge_rrp_available), cfg.vesicle_depletion_fraction
    )
    challenge_rng = np.random.default_rng(structural_seed + 7707)
    eligible_challenge_edges = np.flatnonzero(
        np.isin(edge_mt_route, list(challenged_routes))
    )
    n_challenged = int(round(
        cfg.challenge_route_fraction * len(eligible_challenge_edges)
    ))
    challenged_edges = np.zeros(len(edge_mt_route), dtype=bool)
    if n_challenged:
        challenged_edges[
            challenge_rng.choice(
                eligible_challenge_edges, n_challenged, replace=False
            )
        ] = True
    # Phenomenological use-dependent transmission resource for the reduced Ia
    # and effective Ib spinal interfaces. It is not PAD, a receptor subtype,
    # a presynaptic vesicle pool, or a claim about a monosynaptic Ib->MN path.
    sensory_resource = np.ones(8)
    spatial_edge_map = np.arange(len(edge_rrp_available), dtype=np.int32)
    spatial_rng = np.random.default_rng(structural_seed + 8808)
    # Shuffle every presynaptic source class independently within side. An
    # earlier candidate stopped one route early and left V2b unshuffled.
    for route_id in range(len(MT_ROUTES)):
        for side_id in range(len(SIDES)):
            ids = np.flatnonzero(
                (edge_route_index == route_id)
                & (edge_side_index == side_id)
            )
            if len(ids):
                spatial_edge_map[ids] = spatial_rng.permutation(ids)
    challenge_applied = False

    n_step = expected_n_step
    time_s = np.arange(n_step) * dt / 1000.0
    sample_every = max(1, int(round(1.0 / dt)))
    sample_end_steps = np.arange(
        sample_every, n_step + 1, sample_every, dtype=np.int64
    )
    if not len(sample_end_steps) or sample_end_steps[-1] != n_step:
        sample_end_steps = np.r_[sample_end_steps, n_step]
    n_sample = len(sample_end_steps)
    sample_time = np.empty(n_sample)
    sample_rate = np.empty((n_sample, n_pop))
    sample_mt_pop = np.empty((n_sample, n_pop))
    sample_mt_side = np.empty((n_sample, 2))
    sample_mt_activity_side = np.empty((n_sample, 2))
    sample_mt_route = np.empty((n_sample, len(MT_ROUTES)))
    sample_mt_route_activity = np.empty((n_sample, len(MT_ROUTES)))
    sample_calcium_side = np.empty((n_sample, 2))
    sample_kca_activation_side = np.empty((n_sample, 2))
    sample_rrp_route = np.empty((n_sample, len(MT_ROUTES)))
    sample_replenishment_resource_route = np.empty(
        (n_sample, len(MT_ROUTES))
    )
    sample_ia_signal = np.empty((n_sample, 4))
    sample_ib_signal = np.empty((n_sample, 4))
    sample_ia_transmission = np.empty((n_sample, 4))
    sample_ib_transmission = np.empty((n_sample, 4))
    sample_nmj_vesicle = np.empty((n_sample, 4))
    sample_nmj_ach_gate = np.empty((n_sample, 4))
    sample_nmj_endplate_mv = np.empty((n_sample, 4))
    sample_nmj_nachr_open = np.empty((n_sample, 4))
    sample_muscle_fiber_excitation = np.empty((n_sample, 4))
    sample_muscle_calcium = np.empty((n_sample, 4))
    sample_muscle_activation = np.empty((n_sample, 4))
    sample_muscle_force = np.empty((n_sample, 4))
    sample_joint_state = np.empty((n_sample, 4))
    sample_mn = np.empty((n_sample, 4))
    sample_challenged_rrp = np.empty(n_sample)
    sample_challenged_replenishment_resource = np.empty(n_sample)
    mechanism_component_names = np.asarray([
        "RG_subset_NaP_LCa_M_KCa_Ih_IT_IA_gap",
        "PF_slow_premotor_integration",
        "MN_NAP_LTYPE_CA_PIC_AHP",
        "V2a_variant_Ih_delay_gap", "V3_dorsoventral_Ih_IT_VLat_microcircuit",
        "Renshaw_mixed_MN_input_Ih_SK",
    ])
    mechanism_abs_sum = np.zeros(len(mechanism_component_names))
    renshaw_mixed_input_abs_sum = np.zeros(2)
    mechanism_observation_count = 0
    class_membrane_rhs_abs_sum = np.zeros(len(CLASSES))
    class_synaptic_current_abs_sum = np.zeros(len(CLASSES))
    class_direct_input_current_abs_sum = np.zeros(len(CLASSES))
    class_current_observation_count = 0
    class_spike_count = np.zeros(len(CLASSES), dtype=np.int64)
    class_intrinsic_term_abs_sum = np.zeros((
        len(CLASSES), len(RUNTIME_INTRINSIC_TERM_ORDER)
    ))
    class_intrinsic_term_declared = np.asarray([
        [
            term_id in CLASS_EXECUTION_CONTRACTS[cell_class].intrinsic_term_ids
            for term_id in RUNTIME_INTRINSIC_TERM_ORDER
        ]
        for cell_class in CLASSES
    ], dtype=bool)
    class_direct_input_term_abs_sum = np.zeros((
        len(CLASSES), len(RUNTIME_DIRECT_INPUT_TERM_ORDER)
    ))
    class_direct_input_term_declared = np.asarray([
        [
            term_id in CLASS_EXECUTION_CONTRACTS[cell_class].direct_input_ids
            for term_id in RUNTIME_DIRECT_INPUT_TERM_ORDER
        ]
        for cell_class in CLASSES
    ], dtype=bool)
    sample_idx = 0
    spike_time: List[float] = []
    spike_population: List[int] = []
    spike_neuron: List[int] = []
    terminal_release_event_time: List[float] = []
    terminal_release_event_context: List[int] = []
    terminal_release_event_fraction: List[float] = []
    terminal_release_delivery_time: List[float] = []
    terminal_release_delivery_context: List[int] = []
    terminal_release_delivery_fraction: List[float] = []

    syn_exc_decay = math.exp(-dt / cfg.excitatory_synapse_decay_ms)
    syn_inh_decay = math.exp(-dt / cfg.inhibitory_synapse_decay_ms)
    renshaw_nachr_decay = math.exp(-dt / cfg.renshaw_nachr_decay_ms)
    renshaw_glutamate_decay = math.exp(
        -dt / cfg.renshaw_glutamate_decay_ms
    )
    mt_activity_decay = math.exp(-dt / cfg.mt_activity_tau_ms)
    calcium_decay = math.exp(-dt / cfg.calcium_decay_ms)
    mn_calcium_decay = math.exp(-dt / cfg.mn_calcium_decay_ms)
    mn_nap_pic_inactivation_relaxation = 1.0 - math.exp(
        -dt / cfg.mn_nap_pic_inactivation_tau_ms
    )
    mn_ltype_ca_pic_activation_relaxation = 1.0 - math.exp(
        -dt / cfg.mn_ltype_ca_pic_activation_tau_ms
    )
    v2a_h_relaxation = 1.0 - math.exp(-dt / cfg.v2a_h_tau_ms)
    rg_ltype_ca_pic_activation_relaxation = 1.0 - math.exp(
        -dt / cfg.rg_ltype_ca_pic_activation_tau_ms
    )
    rg_h_relaxation = 1.0 - math.exp(-dt / cfg.rg_h_tau_ms)
    rg_t_inactivation_relaxation = 1.0 - math.exp(
        -dt / cfg.rg_t_inactivation_tau_ms
    )
    rg_a_inactivation_relaxation = 1.0 - math.exp(
        -dt / cfg.rg_a_inactivation_tau_ms
    )
    renshaw_calcium_decay = math.exp(-dt / cfg.renshaw_calcium_decay_ms)
    nmj_vesicle_recovery_decay = math.exp(
        -dt / cfg.nmj_vesicle_recovery_ms
    )
    nmj_ach_decay = math.exp(-dt / cfg.nmj_ach_decay_ms)
    nmj_endplate_decay = math.exp(-dt / cfg.nmj_endplate_tau_ms)
    muscle_activation_decay = math.exp(-dt / cfg.muscle_activation_tau_ms)
    noise_factor = math.sqrt(2.0 * dt / cfg.noise_tau_ms)
    rate_decay = math.exp(-dt / cfg.rate_tau_ms)
    kick_population = pop("RG", "R", "F")
    kick_mask = neuron_pop == kick_population

    route_edge_masks = [edge_route_index == index for index in range(len(MT_ROUTES))]
    population_route_index = np.full(n_pop, -1, dtype=int)
    for p_index, name in enumerate(POPULATIONS):
        cell_class = name.rsplit("_", 2)[0]
        population_route_index[p_index] = route_index[cell_class]

    mt_side_masks = [
        np.asarray([POPULATIONS[p].rsplit("_", 2)[1] == side for p in neuron_pop])
        for side in SIDES
    ]

    pf_delete_mask = neuron_pop == pop("PF", "L", "F")
    pulse_population = pop("RG", pulse_target_side, pulse_target_phase)
    pulse_mask = neuron_pop == pulse_population
    pulse_start_step = -1
    pulse_end_step = -1
    sham_excitatory_start_step = -1
    sham_excitatory_end_step = -1
    sham_inhibitory_start_step = -1
    sham_inhibitory_end_step = -1
    pulse_trigger_time_s = float("nan")
    target_online_onsets: List[float] = []
    target_burst_armed = True
    previous_target_rate = 0.0
    latest_ia_signal = np.zeros(4)
    latest_ib_signal = np.zeros(4)
    latest_ia_transmission = np.zeros(4)
    latest_ib_transmission = np.zeros(4)
    pf_context_masks = tuple(
        neuron_pop == pop("PF", side, phase)
        for side, phase in SIDE_PHASES
    )
    mn_context_masks = tuple(
        neuron_pop == pop("MN", side, phase)
        for side, phase in SIDE_PHASES
    )
    v1ia_context_masks = tuple(
        neuron_pop == pop("V1Ia", side, phase)
        for side, phase in SIDE_PHASES
    )
    rg_context_indices = np.asarray([
        pop("RG", side, phase) for side, phase in SIDE_PHASES
    ], dtype=np.int16)
    mn_context_indices = np.asarray([
        pop("MN", side, phase) for side, phase in SIDE_PHASES
    ], dtype=np.int16)
    # Four explicit NMJs: L-flexor, L-extensor, R-flexor, R-extensor.
    nmj_vesicle_available = np.ones(4)
    nmj_ach_gate = np.zeros(4)
    nmj_endplate_mv = np.full(4, cfg.muscle_fiber_rest_mv)
    nmj_nachr_open = np.zeros(4)
    resting_fiber_gate = float(sigmoid(np.asarray([
        (cfg.muscle_fiber_rest_mv - cfg.muscle_fiber_threshold_mv)
        / cfg.muscle_fiber_slope_mv
    ]))[0])
    muscle_fiber_excitation = np.zeros(4)
    muscle_calcium = np.zeros(4)
    muscle_activation = np.zeros(4)
    muscle_force = np.zeros(4)
    mn_population_context = np.full(n_pop, -1, dtype=np.int8)
    for context_index, (side, phase) in enumerate(SIDE_PHASES):
        mn_population_context[pop("MN", side, phase)] = context_index
    nmj_ring_length = int(math.ceil(cfg.nmj_release_delay_ms / dt)) + 3
    scheduled_nmj_event_fraction = np.zeros((nmj_ring_length, 4))
    nmj_ring_pointer = 0
    nmj_scheduled_event_fraction_total = 0.0
    nmj_arrived_event_fraction_total = 0.0
    v3_motor_scheduled_edge_event_count_by_subphenotype = np.zeros(
        2, dtype=np.int64
    )
    v3_microcircuit_scheduled_edge_event_counts = np.zeros(
        len(v3_microcircuit_pathway_names), dtype=np.int64
    )
    nmj_delay_reconstruction_max_abs_error_ms = 0.0
    nmj_event_split_mass_max_abs_error = 0.0
    joint_position = np.zeros(2)
    joint_velocity = np.zeros(2)
    impaired_route_set = set(impaired_mt_routes)
    impaired_route_mask = np.asarray([
        mt_mode == "impaired" or route in impaired_route_set
        for route in MT_ROUTES
    ])
    edge_route_impaired = impaired_route_mask[
        edge_route_index[resource_edge_ids]
    ]
    edge_build_scale = np.where(
        edge_route_impaired, cfg.mt_impaired_nucleation_scale, 1.0
    )
    edge_decay_ms = np.where(
        edge_route_impaired,
        cfg.mt_track_decay_ms * cfg.mt_impaired_lifetime_scale,
        cfg.mt_track_decay_ms,
    )
    # Draw the identical flat PCG stream in cache-sized blocks.  The former
    # implementation crossed the Python/NumPy boundary twice per integration
    # step; C-order rows preserve the prior per-step neuron-then-population
    # draw ordering exactly.
    noise_block_size = 1024
    noise_block = np.empty((0, n_neuron + n_pop), dtype=float)
    noise_block_cursor = 0

    for step, t_s in enumerate(time_s):
        step_end_s = (step + 1) * dt / 1000.0
        if protocol == "long":
            epoch = min(
                cfg.long_n_epochs,
                int(t_s / cfg.long_epoch_duration_s) + 1,
            )
            demand_active = (
                cfg.long_demand_start_epoch <= epoch <= cfg.long_demand_end_epoch
            )
            challenge_active = epoch >= cfg.long_challenge_epoch
        else:
            epoch = 0
            demand_active = False
            challenge_active = False
        noise_multiplier = (
            cfg.noise_burst_multiplier
            if protocol == "noise_burst" and cfg.perturbation_start_s <= t_s < cfg.perturbation_end_s
            else (cfg.noise_burst_multiplier if demand_active else 1.0)
        )
        if noise_block_cursor >= len(noise_block):
            block_rows = min(noise_block_size, expected_n_step - step)
            noise_block = noise_rng.standard_normal(
                (block_rows, n_neuron + n_pop)
            )
            noise_block_cursor = 0
        noise_draw = noise_block[noise_block_cursor]
        noise_block_cursor += 1
        eta += (
            (dt / cfg.noise_tau_ms) * (-eta)
            + noise_factor * cfg.independent_noise_sigma_pa
            * noise_draw[:n_neuron]
        )
        eta_pop += (
            (dt / cfg.noise_tau_ms) * (-eta_pop)
            + noise_factor * cfg.population_common_noise_sigma_pa
            * noise_draw[n_neuron:]
        )
        mt_edge_activity *= mt_activity_decay
        sensory_drive = np.r_[
            np.zeros(4) if ia_ablated else latest_ia_signal,
            np.zeros(4) if ib_ablated else latest_ib_signal,
        ]
        if mt_mode == "time_yoked" and not mt_time_api:
            mt_event_end = int(np.searchsorted(mt_event_steps, step, side="right"))
            if mt_event_end > mt_event_cursor:
                replay_edges = mt_event_edges[mt_event_cursor:mt_event_end]
                np.add.at(
                    mt_edge_activity,
                    replay_edges,
                    cfg.mt_activity_spike_increment,
                )
            mt_event_cursor = mt_event_end
        calcium *= calcium_decay
        mn_calcium *= mn_calcium_decay
        renshaw_calcium *= renshaw_calcium_decay
        if fast_mode == "yoked" and not kca_time_api:
            kca_event_end = int(np.searchsorted(kca_event_steps, step, side="right"))
            if kca_event_end > kca_event_cursor:
                ids = kca_event_neurons[kca_event_cursor:kca_event_end]
                np.add.at(calcium, ids, cfg.calcium_spike_increment)
            kca_event_cursor = kca_event_end
        if mt_mode not in {"off", "static_matched"}:
            mt_edge_tracks[resource_edge_ids] += dt * (
                -mt_edge_tracks[resource_edge_ids] / edge_decay_ms
                + edge_build_scale * cfg.mt_nucleation_per_ms
                * mt_edge_activity[resource_edge_ids]
                * (1.0 - mt_edge_tracks[resource_edge_ids] / cfg.mt_track_max)
            )
            mt_edge_tracks = np.clip(
                mt_edge_tracks, 0.0, cfg.mt_track_max
            )

        edge_effective_tracks = mt_edge_tracks

        # Exogenous challenge: RRP and the normalized slow-replenishment
        # resource are set to identical floors in every MT condition. The
        # resource is not an anatomical vesicle reserve pool. There is no
        # damage/repair state and no MT term in challenge delivery.
        if challenge_active and not challenge_applied:
            edge_rrp_available[challenged_edges] = cfg.long_rrp_challenge_floor
            edge_slow_replenishment_resource[challenged_edges] = (
                cfg.long_replenishment_resource_challenge_floor
            )
            challenge_applied = True

        # MT affects only a slow RRP-replenishment component gated by a
        # normalized phenomenological resource. Richards et al. support RRP
        # recycling/refill, not anatomical reserve-pool mobilization; the
        # resource depletion/recovery realization below is explicitly H-level.
        # The fast component and resource recovery are MT-independent.
        fast_flux = (
            1.0 - edge_rrp_available[resource_edge_ids]
        ) / cfg.vesicle_fast_recovery_ms
        slow_flux = (
            (1.0 + cfg.mt_slow_replenishment_gain
             * edge_effective_tracks[resource_edge_ids])
            * edge_slow_replenishment_resource[resource_edge_ids]
            * (1.0 - edge_rrp_available[resource_edge_ids])
            / cfg.vesicle_slow_recovery_ms
        )
        edge_rrp_available[resource_edge_ids] += dt * (fast_flux + slow_flux)
        edge_slow_replenishment_resource[resource_edge_ids] += dt * (
            (1.0 - edge_slow_replenishment_resource[resource_edge_ids])
            / cfg.slow_replenishment_resource_recovery_ms
            - slow_flux
        )
        edge_rrp_available[resource_edge_ids] = np.clip(
            edge_rrp_available[resource_edge_ids], 0.0, 1.0
        )
        edge_slow_replenishment_resource[resource_edge_ids] = np.clip(
            edge_slow_replenishment_resource[resource_edge_ids], 0.0, 1.0
        )
        sensory_resource += dt * (
            1.0 - sensory_resource
        ) / cfg.sensory_resource_recovery_ms
        sensory_resource -= (
            dt / 1000.0 * cfg.sensory_resource_depletion_per_s
            * (1.35 if demand_active else 1.0)
            * (sensory_drive > 0.0)
        )
        sensory_resource = np.clip(sensory_resource, 0.0, 1.0)

        syn_exc_conductance *= syn_exc_decay
        syn_exc_conductance += scheduled_exc_conductance[ring_pointer]
        scheduled_exc_conductance[ring_pointer].fill(0.0)
        syn_inh_conductance *= syn_inh_decay
        syn_inh_conductance += scheduled_inh_conductance[ring_pointer]
        scheduled_inh_conductance[ring_pointer].fill(0.0)
        renshaw_nachr_conductance *= renshaw_nachr_decay
        renshaw_nachr_conductance += scheduled_nachr_conductance[ring_pointer]
        scheduled_nachr_conductance[ring_pointer].fill(0.0)
        renshaw_glutamate_conductance *= renshaw_glutamate_decay
        renshaw_glutamate_conductance += (
            scheduled_renshaw_glutamate_conductance[ring_pointer]
        )
        scheduled_renshaw_glutamate_conductance[ring_pointer].fill(0.0)

        i_syn_exc = syn_exc_conductance * (cfg.excitatory_reversal_mv - v)
        i_syn_inh = syn_inh_conductance * (cfg.inhibitory_reversal_mv - v)
        i_renshaw_nachr = (
            renshaw_nachr_conductance * (cfg.excitatory_reversal_mv - v)
        )
        i_renshaw_glutamate = (
            renshaw_glutamate_conductance
            * (cfg.excitatory_reversal_mv - v)
        )
        i_syn = syn_scale * (
            i_syn_exc + i_syn_inh
            + i_renshaw_nachr + i_renshaw_glutamate
        )
        drive_now = tonic_drive * drive_scale
        if protocol == "speed_step":
            drive_now = drive_now.copy()
            requested_index = 0 if t_s < cfg.perturbation_start_s else 2
            drive_now[is_rg] += (
                cfg.descending_rg_drive_offsets_pa[requested_index]
                - cfg.descending_rg_drive_offsets_pa[speed_index]
            ) * drive_scale[is_rg]
        tonic_class_current = base_tonic_drive * drive_scale
        descending_rg_current = drive_now - tonic_class_current
        perturbation_current = np.zeros(n_neuron)
        pf_deletion_current = np.zeros(n_neuron)
        if (
            protocol == "phase_kick"
            and cfg.perturbation_start_s <= t_s
            < cfg.perturbation_start_s + cfg.phase_kick_duration_ms / 1000.0
        ):
            perturbation_current[kick_mask] = cfg.phase_kick_current_pa
        if protocol == "pf_deletion" and cfg.perturbation_start_s <= t_s < cfg.perturbation_end_s:
            pf_deletion_current[pf_delete_mask] = cfg.pf_deletion_current_pa
        if (
            protocol == "pulse" and pulse_direction != "none"
            and pulse_start_step <= step < pulse_end_step
        ):
            pulse_current = (
                cfg.excitatory_pulse_current_pa
                if pulse_direction == "excitatory"
                else cfg.inhibitory_pulse_current_pa
            )
            perturbation_current[pulse_mask] = pulse_current
        kick_current = perturbation_current + pf_deletion_current

        # Explicit population NMJ. Terminal-arrived MN event fractions are
        # delivered through a fixed physical-delay ring; the latency therefore
        # does not shrink when the numerical step is refined.
        sensory_current = np.zeros(n_neuron)
        ia_to_pf_current = np.zeros(n_neuron)
        ia_to_mn_current = np.zeros(n_neuron)
        ia_to_v1ia_current = np.zeros(n_neuron)
        ib_to_pf_effective_current = np.zeros(n_neuron)
        ib_to_mn_effective_current = np.zeros(n_neuron)
        nmj_event_fraction = scheduled_nmj_event_fraction[
            nmj_ring_pointer
        ].copy()
        scheduled_nmj_event_fraction[nmj_ring_pointer].fill(0.0)
        nmj_arrived_event_fraction_total += float(np.sum(nmj_event_fraction))
        arrived_contexts = np.flatnonzero(nmj_event_fraction > 0.0)
        if len(arrived_contexts):
            terminal_release_delivery_time.extend(
                [float(t_s)] * len(arrived_contexts)
            )
            terminal_release_delivery_context.extend(arrived_contexts.tolist())
            terminal_release_delivery_fraction.extend(
                nmj_event_fraction[arrived_contexts].tolist()
            )
        nmj_released = np.minimum(
            nmj_vesicle_available,
            cfg.nmj_release_probability * nmj_vesicle_available
            * np.maximum(nmj_event_fraction, 0.0),
        )
        nmj_vesicle_available -= nmj_released
        nmj_vesicle_available = 1.0 - (
            1.0 - nmj_vesicle_available
        ) * nmj_vesicle_recovery_decay
        nmj_vesicle_available = np.clip(
            nmj_vesicle_available, 0.0, 1.0
        )
        nmj_ach_gate += (
            cfg.nmj_ach_release_gain * nmj_released * (1.0 - nmj_ach_gate)
        )
        nmj_ach_gate = np.clip(nmj_ach_gate, 0.0, 1.0) * nmj_ach_decay
        ach_power = np.power(nmj_ach_gate, cfg.nmj_nachr_hill)
        nmj_nachr_open = ach_power / (
            ach_power + cfg.nmj_nachr_half_ach ** cfg.nmj_nachr_hill
        )
        nmj_endplate_target_mv = (
            cfg.muscle_fiber_rest_mv
            + cfg.nmj_endplate_gain_mv * nmj_nachr_open
        )
        nmj_endplate_mv = nmj_endplate_target_mv + (
            nmj_endplate_mv - nmj_endplate_target_mv
        ) * nmj_endplate_decay
        muscle_fiber_excitation_raw = sigmoid(
            (nmj_endplate_mv - cfg.muscle_fiber_threshold_mv)
            / cfg.muscle_fiber_slope_mv
        )
        muscle_fiber_excitation = np.clip(
            (muscle_fiber_excitation_raw - resting_fiber_gate)
            / (1.0 - resting_fiber_gate),
            0.0, 1.0,
        )
        effective_sr_release = (
            cfg.muscle_effective_sr_release_per_ms * muscle_fiber_excitation
        )
        calcium_relaxation_rate = (
            effective_sr_release + 1.0 / cfg.muscle_calcium_tau_ms
        )
        calcium_steady_state = effective_sr_release / calcium_relaxation_rate
        muscle_calcium = calcium_steady_state + (
            muscle_calcium - calcium_steady_state
        ) * np.exp(-dt * calcium_relaxation_rate)
        muscle_calcium = np.clip(muscle_calcium, 0.0, 1.0)
        calcium_power = np.power(
            muscle_calcium, cfg.muscle_calcium_hill
        )
        calcium_activation = calcium_power / (
            calcium_power + cfg.muscle_calcium_half ** cfg.muscle_calcium_hill
        )
        muscle_activation = calcium_activation + (
            muscle_activation - calcium_activation
        ) * muscle_activation_decay
        load_resistance_factors = np.ones(2)
        if load_context == "unilateral":
            load_resistance_factors[
                SIDES.index(load_side)
            ] = cfg.load_unilateral_resistance_multiplier
        elif load_context == "bilateral_high":
            load_resistance_factors.fill(
                cfg.load_bilateral_high_resistance_multiplier
            )
        if demand_active:
            load_resistance_factors *= (
                cfg.load_bilateral_high_resistance_multiplier
            )
        previous_joint = joint_position.copy()
        for side_id in range(2):
            flexor_id = 2 * side_id
            extensor_id = flexor_id + 1
            muscle_force[flexor_id] = muscle_activation[flexor_id]
            muscle_force[extensor_id] = (
                muscle_activation[extensor_id] * cfg.extensor_force_scale_prior
            )
            torque = (
                cfg.muscle_torque_gain
                * (muscle_force[flexor_id] - muscle_force[extensor_id])
                - cfg.joint_damping * load_resistance_factors[side_id]
                * joint_velocity[side_id]
                - cfg.joint_stiffness * joint_position[side_id]
            )
            joint_velocity[side_id] += (
                dt / 1000.0 * torque / cfg.joint_inertia
            )
            joint_position[side_id] += dt / 1000.0 * joint_velocity[side_id]
        joint_velocity_observed = (
            joint_position - previous_joint
        ) / max(dt / 1000.0, 1e-12)
        muscle_length = np.asarray([
            1.0 - cfg.muscle_length_scale * joint_position[0],
            1.0 + cfg.muscle_length_scale * joint_position[0],
            1.0 - cfg.muscle_length_scale * joint_position[1],
            1.0 + cfg.muscle_length_scale * joint_position[1],
        ])
        muscle_length_velocity = np.asarray([
            -cfg.muscle_length_scale * joint_velocity_observed[0],
            cfg.muscle_length_scale * joint_velocity_observed[0],
            -cfg.muscle_length_scale * joint_velocity_observed[1],
            cfg.muscle_length_scale * joint_velocity_observed[1],
        ])
        # First Ia interface: a bounded spindle length/velocity proxy. It has
        # no central target or synaptic gain by itself.
        latest_ia_signal = np.clip(
            cfg.ia_tonic
            + cfg.ia_length_gain * np.maximum(muscle_length - 1.0, 0.0)
            + cfg.ia_velocity_gain * np.maximum(muscle_length_velocity, 0.0),
            0.0, 2.5,
        )
        # First interface: GTO-like force transduction only. Its output is a
        # bounded proxy and has no central synaptic sign by itself.
        latest_ib_signal = np.clip(
            cfg.ib_tonic + cfg.ib_force_gain * muscle_force,
            0.0, 2.5,
        )
        # Second Ia interface: a reduced use-dependent spinal transmission
        # factor and effective target currents. Ia->MN/IaIN circuit motifs
        # have direct support; the PF target and exact gains remain H-level.
        latest_ia_transmission = (
            np.zeros(4) if ia_ablated
            else latest_ia_signal * sensory_resource[:4]
        )
        latest_ib_transmission = (
            np.zeros(4) if ib_ablated
            else latest_ib_signal * sensory_resource[4:]
        )
        if not ia_ablated:
            for context_index, (side, phase) in enumerate(SIDE_PHASES):
                afferent = latest_ia_transmission[context_index]
                pf_mask = pf_context_masks[context_index]
                mn_mask = mn_context_masks[context_index]
                v1ia_mask = v1ia_context_masks[context_index]
                ia_to_pf_current[pf_mask] = cfg.ia_to_pf_pa * afferent
                ia_to_mn_current[mn_mask] = cfg.ia_to_mn_pa * afferent
                ia_to_v1ia_current[v1ia_mask] = cfg.ia_to_v1ia_pa * afferent
                sensory_current[pf_mask] += ia_to_pf_current[pf_mask]
                sensory_current[mn_mask] += ia_to_mn_current[mn_mask]
                sensory_current[v1ia_mask] += ia_to_v1ia_current[v1ia_mask]
        # Second interface: unresolved effective locomotor spinal action. The
        # fixed positive sign and extensor context ratio are H-level priors for
        # this represented regime; phase/context-dependent reflex reversal and
        # anatomical monosynaptic Ib excitation are not modeled.
        if not ib_ablated:
            for context_index, (side, phase) in enumerate(SIDE_PHASES):
                gain = (
                    cfg.ib_effective_extensor_context_gain
                    if phase == "E" else 1.0
                )
                afferent = latest_ib_transmission[context_index] * gain
                pf_mask = pf_context_masks[context_index]
                mn_mask = mn_context_masks[context_index]
                ib_to_pf_effective_current[pf_mask] = (
                    cfg.ib_effective_spinal_to_pf_pa * afferent
                )
                ib_to_mn_effective_current[mn_mask] = (
                    cfg.ib_effective_spinal_to_mn_pa * afferent
                )
                sensory_current[pf_mask] += ib_to_pf_effective_current[pf_mask]
                sensory_current[mn_mask] += ib_to_mn_effective_current[mn_mask]

        # PF slow integration is a functional premotor mechanism, not a claim
        # of a PF-specific molecular identity.
        pf_synaptic_activation = np.maximum(i_syn_exc, 0.0) / (
            cfg.pf_slow_integration_half_pa + np.maximum(i_syn_exc, 0.0)
        )
        pf_slow_gate += dt * (
            pf_synaptic_activation * (1.0 - pf_slow_gate)
            / cfg.pf_slow_integration_rise_ms
            - pf_slow_gate / cfg.pf_slow_integration_decay_ms
        )
        pf_slow_gate = np.clip(pf_slow_gate, 0.0, 1.0)
        i_pf_slow = (
            cfg.pf_slow_integration_conductance_ns * is_pf * pf_slow_gate
            * (cfg.excitatory_reversal_mv - v)
        )

        # V2a delayed-onset subset: an effective transient outward term fitted
        # to the measured delayed firing phenotype. It is intentionally not
        # assigned an A-channel molecular identity. Electrical coupling uses
        # separate tonic/phasic pair graphs at their measured pair incidences;
        # delayed pairs were not sampled and therefore are not represented.
        v2a_delay_activation = sigmoid(
            (v - cfg.v2a_delayed_activation_half_mv)
            / cfg.v2a_delayed_activation_slope_mv
        )
        v2a_delay_relief += dt * (
            v2a_delay_activation - v2a_delay_relief
        ) / cfg.v2a_delayed_relief_tau_ms
        v2a_delay_relief = np.clip(v2a_delay_relief, 0.0, 1.0)
        v2a_delayed_mask = v2a_variant == 2
        i_v2a_delay = (
            cfg.v2a_delayed_onset_conductance_ns * v2a_delayed_mask
            * v2a_delay_activation * (1.0 - v2a_delay_relief)
            * (cfg.potassium_reversal_mv - v)
        )
        i_v2a_gap_tonic = symmetric_pair_gap_current(
            v, v2a_tonic_gap_source, v2a_tonic_gap_target,
            cfg.v2a_gap_conductance_ns,
        )
        i_v2a_gap_phasic = symmetric_pair_gap_current(
            v, v2a_phasic_gap_source, v2a_phasic_gap_target,
            cfg.v2a_gap_conductance_ns,
        )
        v2a_h_inf = sigmoid(-(v - cfg.v2a_h_half_mv) / cfg.v2a_h_slope_mv)
        v2a_h_gate += v2a_h_relaxation * (v2a_h_inf - v2a_h_gate)
        v2a_h_common = (
            cfg.v2a_h_conductance_ns * v2a_h_gate
            * (cfg.v2a_h_reversal_mv - v)
        )
        i_v2a_h_tonic = (
            v2a_h_common * v2a_h_positive_mask * (v2a_variant == 0)
        )
        i_v2a_h_phasic = (
            v2a_h_common * v2a_h_positive_mask * (v2a_variant == 1)
        )
        i_v2a_h_delayed = (
            v2a_h_common * v2a_h_positive_mask * (v2a_variant == 2)
        )

        # Both V3 phenotypes express Ih; the dorsal-like group has the larger
        # effective conductance and alone receives the T-type rebound current.
        v3_h_inf = sigmoid(-(v - cfg.v3_h_half_mv) / cfg.v3_h_slope_mv)
        v3_h_gate += dt * (v3_h_inf - v3_h_gate) / cfg.v3_h_tau_ms
        v3_ventral_mask = is_v3 & ~v3_dorsal_mask
        i_v3_h_ventral = (
            cfg.v3_ventral_h_conductance_ns * v3_ventral_mask * v3_h_gate
            * (cfg.v3_h_reversal_mv - v)
        )
        i_v3_h_dorsal = (
            cfg.v3_ventral_h_conductance_ns
            * cfg.v3_dorsal_h_conductance_multiplier
            * v3_dorsal_mask * v3_h_gate
            * (cfg.v3_h_reversal_mv - v)
        )
        v3_t_activation = sigmoid(
            (v - cfg.v3_t_activation_half_mv) / cfg.v3_t_activation_slope_mv
        )
        v3_t_inactivation_inf = sigmoid(
            -(v - cfg.v3_t_inactivation_half_mv)
            / cfg.v3_t_inactivation_slope_mv
        )
        v3_t_inactivation += dt * (
            v3_t_inactivation_inf - v3_t_inactivation
        ) / cfg.v3_t_inactivation_tau_ms
        i_v3_t_dorsal = (
            cfg.v3_t_conductance_ns * v3_dorsal_mask
            * v3_t_activation ** 2 * v3_t_inactivation
            * (cfg.v3_t_reversal_mv - v)
        )

        # Chrna2-positive Renshaw-cell Ih and SK currents. The nAChR current is
        # carried separately from generic central synapses above.
        renshaw_h_inf = sigmoid(
            -(v - cfg.renshaw_h_half_mv) / cfg.renshaw_h_slope_mv
        )
        renshaw_h_gate += dt * (
            renshaw_h_inf - renshaw_h_gate
        ) / cfg.renshaw_h_tau_ms
        i_renshaw_h = (
            cfg.renshaw_h_conductance_ns * is_renshaw * renshaw_h_gate
            * (cfg.renshaw_h_reversal_mv - v)
        )
        renshaw_sk_activation = renshaw_calcium / (
            renshaw_calcium + cfg.renshaw_calcium_half_activation + 1e-12
        )
        i_renshaw_sk = (
            cfg.renshaw_sk_conductance_ns * is_renshaw
            * renshaw_sk_activation * (cfg.potassium_reversal_mv - v)
        )

        m_inf = sigmoid((v - cfg.nap_activation_half_mv) / cfg.nap_activation_slope_mv)
        h_inf = sigmoid(-(v - cfg.nap_inactivation_half_mv) / cfg.nap_inactivation_slope_mv)
        h += dt * (h_inf - h) / cfg.nap_inactivation_tau_ms
        i_nap = nap_conductance * nap_scale * m_inf * h * (cfg.sodium_reversal_mv - v)
        rg_ltype_ca_pic_activation_inf = sigmoid(
            (v - cfg.rg_ltype_ca_pic_activation_half_mv)
            / cfg.rg_ltype_ca_pic_activation_slope_mv
        )
        rg_ltype_ca_pic_activation += (
            rg_ltype_ca_pic_activation_relaxation
            * (rg_ltype_ca_pic_activation_inf - rg_ltype_ca_pic_activation)
        )
        i_rg_ltype_ca_pic = (
            cfg.rg_ltype_ca_pic_conductance_ns * rg_pic_positive_mask
            * rg_ltype_ca_pic_activation
            * (cfg.rg_ltype_ca_reversal_mv - v)
        )
        rg_h_inf = sigmoid(-(v - cfg.rg_h_half_mv) / cfg.rg_h_slope_mv)
        rg_h_gate += rg_h_relaxation * (rg_h_inf - rg_h_gate)
        i_rg_h = (
            cfg.rg_h_conductance_ns * rg_h_positive_mask * rg_h_gate
            * (cfg.rg_h_reversal_mv - v)
        )
        rg_t_activation = sigmoid(
            (v - cfg.rg_t_activation_half_mv) / cfg.rg_t_activation_slope_mv
        )
        rg_t_inactivation_inf = sigmoid(
            -(v - cfg.rg_t_inactivation_half_mv)
            / cfg.rg_t_inactivation_slope_mv
        )
        rg_t_inactivation += rg_t_inactivation_relaxation * (
            rg_t_inactivation_inf - rg_t_inactivation
        )
        i_rg_t = (
            cfg.rg_t_conductance_ns * rg_t_positive_mask
            * rg_t_activation ** 2 * rg_t_inactivation
            * (cfg.rg_t_reversal_mv - v)
        )
        rg_a_activation = sigmoid(
            (v - cfg.rg_a_activation_half_mv) / cfg.rg_a_activation_slope_mv
        )
        rg_a_inactivation_inf = sigmoid(
            -(v - cfg.rg_a_inactivation_half_mv)
            / cfg.rg_a_inactivation_slope_mv
        )
        rg_a_inactivation += rg_a_inactivation_relaxation * (
            rg_a_inactivation_inf - rg_a_inactivation
        )
        i_rg_a = (
            cfg.rg_a_conductance_ns * rg_a_positive_mask
            * rg_a_activation ** 3 * rg_a_inactivation
            * (cfg.potassium_reversal_mv - v)
        )
        i_rg_gap = symmetric_pair_gap_current(
            v, rg_gap_source, rg_gap_target, cfg.rg_gap_conductance_ns
        )
        m_inf_gate = sigmoid((v - cfg.m_activation_half_mv) / cfg.m_activation_slope_mv)
        m_gate += dt * (m_inf_gate - m_gate) / cfg.m_activation_tau_ms
        i_m = (
            cfg.rg_m_conductance_ns * cfg.rg_m_conductance_scale
            * rg_m_positive_mask * m_gate * (cfg.potassium_reversal_mv - v)
        )
        calcium_hill = np.power(np.maximum(calcium, 0.0), cfg.calcium_hill_coefficient)
        kca_activation = calcium_hill / (
            calcium_hill + cfg.calcium_half_activation ** cfg.calcium_hill_coefficient
        )
        if fast_mode == "off":
            kca_activation.fill(0.0)
        elif fast_mode == "static_mean":
            # Mean-matched non-adaptive control; reference is calibrated only on
            # calibration seeds and then frozen for held-out validation.
            kca_activation[is_rg] = cfg.static_kca_activation_reference
        elif fast_mode == "yoked":
            kca_activation *= fast_activation_scale
        i_kca = (
            cfg.kca_conductance_ns * rg_kca_positive_mask * kca_activation
            * (cfg.potassium_reversal_mv - v)
        )
        mn_nap_pic_activation = sigmoid(
            (mn_dendrite_v - cfg.mn_nap_pic_activation_half_mv)
            / cfg.mn_nap_pic_activation_slope_mv
        )
        mn_nap_pic_inactivation_inf = sigmoid(
            -(mn_dendrite_v - cfg.mn_nap_pic_inactivation_half_mv)
            / cfg.mn_nap_pic_inactivation_slope_mv
        )
        mn_nap_pic_inactivation += mn_nap_pic_inactivation_relaxation * (
            mn_nap_pic_inactivation_inf - mn_nap_pic_inactivation
        )
        mn_ltype_ca_pic_activation_inf = sigmoid(
            (mn_dendrite_v - cfg.mn_ltype_ca_pic_activation_half_mv)
            / cfg.mn_ltype_ca_pic_activation_slope_mv
        )
        mn_ltype_ca_pic_activation += mn_ltype_ca_pic_activation_relaxation * (
            mn_ltype_ca_pic_activation_inf - mn_ltype_ca_pic_activation
        )
        i_mn_nap_pic = (
            cfg.mn_nap_pic_conductance_ns * is_mn * mn_nap_pic_activation
            * mn_nap_pic_inactivation
            * (cfg.sodium_reversal_mv - mn_dendrite_v)
        )
        i_mn_ltype_ca_pic = (
            cfg.mn_ltype_ca_pic_conductance_ns * is_mn
            * mn_ltype_ca_pic_activation
            * (cfg.mn_ltype_ca_reversal_mv - mn_dendrite_v)
        )
        i_mn_coupling_soma = (
            cfg.mn_soma_dendrite_coupling_ns * is_mn
            * (mn_dendrite_v - v)
        )
        i_mn_coupling_dendrite = -i_mn_coupling_soma
        mn_dendritic_synaptic_current = (
            cfg.mn_dendritic_synaptic_fraction * is_mn
            * np.maximum(i_syn + sensory_current, 0.0)
        )
        i_mn_dendrite_leak = (
            -cfg.mn_dendrite_leak_ns * is_mn
            * (mn_dendrite_v - cfg.leak_reversal_mv)
        )
        mn_ahp_activation = mn_calcium / (
            mn_calcium + cfg.mn_calcium_half_activation + 1e-12
        )
        i_mn_ahp = (
            cfg.mn_ahp_conductance_ns * is_mn * mn_ahp_activation
            * (cfg.potassium_reversal_mv - v)
        )
        exp_term = leak * cfg.slope_factor_mv * np.exp(
            np.clip((v - cfg.threshold_mv) / cfg.slope_factor_mv, -30.0, 12.0)
        )
        i_leak = -leak * (v - cfg.leak_reversal_mv)
        # Local own-voltage derivatives for the exponential Rosenbrock
        # integrator.  Every derivative is taken from the current equation
        # above with its slow gate frozen over this outer step.  Instantaneous
        # activation gates retain their analytic voltage derivative.
        d_m_inf_dv = (
            m_inf * (1.0 - m_inf) / cfg.nap_activation_slope_mv
        )
        d_i_nap_dv = nap_conductance * nap_scale * h * (
            d_m_inf_dv * (cfg.sodium_reversal_mv - v) - m_inf
        )
        d_i_rg_ltype_ca_pic_dv = -(
            cfg.rg_ltype_ca_pic_conductance_ns * rg_pic_positive_mask
            * rg_ltype_ca_pic_activation
        )
        d_i_m_dv = -(
            cfg.rg_m_conductance_ns * cfg.rg_m_conductance_scale
            * rg_m_positive_mask * m_gate
        )
        d_i_kca_dv = -(
            cfg.kca_conductance_ns * rg_kca_positive_mask * kca_activation
        )
        d_i_rg_h_dv = -(
            cfg.rg_h_conductance_ns * rg_h_positive_mask * rg_h_gate
        )
        d_rg_t_activation_dv = (
            rg_t_activation * (1.0 - rg_t_activation)
            / cfg.rg_t_activation_slope_mv
        )
        d_i_rg_t_dv = (
            cfg.rg_t_conductance_ns * rg_t_positive_mask
            * rg_t_inactivation
            * (
                2.0 * rg_t_activation * d_rg_t_activation_dv
                * (cfg.rg_t_reversal_mv - v)
                - rg_t_activation ** 2
            )
        )
        d_rg_a_activation_dv = (
            rg_a_activation * (1.0 - rg_a_activation)
            / cfg.rg_a_activation_slope_mv
        )
        d_i_rg_a_dv = (
            cfg.rg_a_conductance_ns * rg_a_positive_mask
            * rg_a_inactivation
            * (
                3.0 * rg_a_activation ** 2 * d_rg_a_activation_dv
                * (cfg.potassium_reversal_mv - v)
                - rg_a_activation ** 3
            )
        )
        d_i_rg_gap_dv = -cfg.rg_gap_conductance_ns * rg_gap_degree
        d_i_pf_slow_dv = -(
            cfg.pf_slow_integration_conductance_ns * is_pf * pf_slow_gate
        )
        d_v2a_delay_activation_dv = (
            v2a_delay_activation * (1.0 - v2a_delay_activation)
            / cfg.v2a_delayed_activation_slope_mv
        )
        d_i_v2a_delay_dv = (
            cfg.v2a_delayed_onset_conductance_ns * v2a_delayed_mask
            * (1.0 - v2a_delay_relief)
            * (
                d_v2a_delay_activation_dv
                * (cfg.potassium_reversal_mv - v)
                - v2a_delay_activation
            )
        )
        d_i_v2a_gap_tonic_dv = -(
            cfg.v2a_gap_conductance_ns * v2a_tonic_gap_degree
        )
        d_i_v2a_gap_phasic_dv = -(
            cfg.v2a_gap_conductance_ns * v2a_phasic_gap_degree
        )
        d_i_v2a_h_tonic_dv = -(
            cfg.v2a_h_conductance_ns * v2a_h_gate
            * v2a_h_positive_mask * (v2a_variant == 0)
        )
        d_i_v2a_h_phasic_dv = -(
            cfg.v2a_h_conductance_ns * v2a_h_gate
            * v2a_h_positive_mask * (v2a_variant == 1)
        )
        d_i_v2a_h_delayed_dv = -(
            cfg.v2a_h_conductance_ns * v2a_h_gate
            * v2a_h_positive_mask * (v2a_variant == 2)
        )
        d_i_v3_h_ventral_dv = -(
            cfg.v3_ventral_h_conductance_ns * v3_ventral_mask * v3_h_gate
        )
        d_i_v3_h_dorsal_dv = -(
            cfg.v3_ventral_h_conductance_ns
            * cfg.v3_dorsal_h_conductance_multiplier
            * v3_dorsal_mask * v3_h_gate
        )
        d_v3_t_activation_dv = (
            v3_t_activation * (1.0 - v3_t_activation)
            / cfg.v3_t_activation_slope_mv
        )
        d_i_v3_t_dorsal_dv = (
            cfg.v3_t_conductance_ns * v3_dorsal_mask * v3_t_inactivation
            * (
                2.0 * v3_t_activation * d_v3_t_activation_dv
                * (cfg.v3_t_reversal_mv - v)
                - v3_t_activation ** 2
            )
        )
        d_i_renshaw_h_dv = -(
            cfg.renshaw_h_conductance_ns * is_renshaw * renshaw_h_gate
        )
        d_i_renshaw_sk_dv = -(
            cfg.renshaw_sk_conductance_ns * is_renshaw
            * renshaw_sk_activation
        )
        d_i_mn_coupling_soma_dv = -(
            cfg.mn_soma_dendrite_coupling_ns * is_mn
        )
        d_i_mn_ahp_dv = -(
            cfg.mn_ahp_conductance_ns * is_mn * mn_ahp_activation
        )
        d_i_syn_dv = -syn_scale * (
            syn_exc_conductance + syn_inh_conductance
            + renshaw_nachr_conductance
            + renshaw_glutamate_conductance
        )
        d_mn_dendritic_synaptic_current_dv = (
            cfg.mn_dendritic_synaptic_fraction * is_mn
            * ((i_syn + sensory_current) > 0.0) * d_i_syn_dv
        )
        d_mn_nap_pic_activation_dv = (
            mn_nap_pic_activation * (1.0 - mn_nap_pic_activation)
            / cfg.mn_nap_pic_activation_slope_mv
        )
        d_i_mn_nap_pic_dvd = (
            cfg.mn_nap_pic_conductance_ns * is_mn
            * mn_nap_pic_inactivation
            * (
                d_mn_nap_pic_activation_dv
                * (cfg.sodium_reversal_mv - mn_dendrite_v)
                - mn_nap_pic_activation
            )
        )
        d_i_mn_ltype_ca_pic_dvd = -(
            cfg.mn_ltype_ca_pic_conductance_ns * is_mn
            * mn_ltype_ca_pic_activation
        )
        d_i_mn_coupling_dendrite_dvd = -(
            cfg.mn_soma_dendrite_coupling_ns * is_mn
        )
        d_i_mn_dendrite_leak_dvd = -cfg.mn_dendrite_leak_ns * is_mn
        # Explicit audit-only causal intervention. Required mechanisms cannot
        # be silently removed through Config; they can be zeroed only through
        # this named, provenance-recorded intervention API. The default empty
        # set leaves the production equations bit-for-bit unchanged.
        if disabled_intrinsic_mechanism_set:
            disablable_currents = {
                "I_NAP_RG": i_nap,
                "I_LTYPE_CA_PIC_RG": i_rg_ltype_ca_pic,
                "I_M_RG": i_m,
                "I_KCA_RG": i_kca,
                "I_H_RG": i_rg_h,
                "I_T_RG": i_rg_t,
                "I_A_RG": i_rg_a,
                "I_RG_GAP": i_rg_gap,
                "I_PF_SLOW": i_pf_slow,
                "I_MN_DEND_LEAK": i_mn_dendrite_leak,
                "I_MN_NAP_PIC": i_mn_nap_pic,
                "I_MN_LTYPE_CA_PIC": i_mn_ltype_ca_pic,
                "I_MN_COUPLING_SOMA": i_mn_coupling_soma,
                "I_MN_AHP": i_mn_ahp,
                "I_V2A_H_TONIC": i_v2a_h_tonic,
                "I_V2A_H_PHASIC": i_v2a_h_phasic,
                "I_V2A_H_DELAYED": i_v2a_h_delayed,
                "I_V2A_DELAY": i_v2a_delay,
                "I_V2A_GAP_TONIC": i_v2a_gap_tonic,
                "I_V2A_GAP_PHASIC": i_v2a_gap_phasic,
                "I_V3_H_VENTRAL": i_v3_h_ventral,
                "I_V3_H_DORSAL": i_v3_h_dorsal,
                "I_V3_T_DORSAL": i_v3_t_dorsal,
                "I_RENSHAW_H": i_renshaw_h,
                "I_RENSHAW_SK": i_renshaw_sk,
            }
            disablable_current_derivatives = {
                "I_NAP_RG": d_i_nap_dv,
                "I_LTYPE_CA_PIC_RG": d_i_rg_ltype_ca_pic_dv,
                "I_M_RG": d_i_m_dv,
                "I_KCA_RG": d_i_kca_dv,
                "I_H_RG": d_i_rg_h_dv,
                "I_T_RG": d_i_rg_t_dv,
                "I_A_RG": d_i_rg_a_dv,
                "I_RG_GAP": d_i_rg_gap_dv,
                "I_PF_SLOW": d_i_pf_slow_dv,
                "I_MN_DEND_LEAK": d_i_mn_dendrite_leak_dvd,
                "I_MN_NAP_PIC": d_i_mn_nap_pic_dvd,
                "I_MN_LTYPE_CA_PIC": d_i_mn_ltype_ca_pic_dvd,
                "I_MN_COUPLING_SOMA": d_i_mn_coupling_soma_dv,
                "I_MN_AHP": d_i_mn_ahp_dv,
                "I_V2A_H_TONIC": d_i_v2a_h_tonic_dv,
                "I_V2A_H_PHASIC": d_i_v2a_h_phasic_dv,
                "I_V2A_H_DELAYED": d_i_v2a_h_delayed_dv,
                "I_V2A_DELAY": d_i_v2a_delay_dv,
                "I_V2A_GAP_TONIC": d_i_v2a_gap_tonic_dv,
                "I_V2A_GAP_PHASIC": d_i_v2a_gap_phasic_dv,
                "I_V3_H_VENTRAL": d_i_v3_h_ventral_dv,
                "I_V3_H_DORSAL": d_i_v3_h_dorsal_dv,
                "I_V3_T_DORSAL": d_i_v3_t_dorsal_dv,
                "I_RENSHAW_H": d_i_renshaw_h_dv,
                "I_RENSHAW_SK": d_i_renshaw_sk_dv,
            }
            for term_id in disabled_intrinsic_mechanism_set:
                disablable_currents[term_id].fill(0.0)
                disablable_current_derivatives[term_id].fill(0.0)
                if term_id == "I_MN_COUPLING_SOMA":
                    i_mn_coupling_dendrite.fill(0.0)
                    d_i_mn_coupling_dendrite_dvd.fill(0.0)
        i_v3_h = i_v3_h_ventral + i_v3_h_dorsal
        i_v3_t = i_v3_t_dorsal
        i_v2a_h = i_v2a_h_tonic + i_v2a_h_phasic + i_v2a_h_delayed
        mn_dendrite_rhs_pa = (
            i_mn_dendrite_leak + i_mn_nap_pic + i_mn_ltype_ca_pic
            + i_mn_coupling_dendrite
            + mn_dendritic_synaptic_current
        )
        mn_dendrite_jacobian_per_ms = (
            d_i_mn_dendrite_leak_dvd
            + d_i_mn_nap_pic_dvd
            + d_i_mn_ltype_ca_pic_dvd
            + d_i_mn_coupling_dendrite_dvd
        ) / cfg.mn_dendrite_capacitance_pf
        d_mn_dendrite_v, _ = exponential_rosenbrock_increment(
            mn_dendrite_rhs_pa / cfg.mn_dendrite_capacitance_pf,
            mn_dendrite_jacobian_per_ms,
            dt,
        )
        direct_input_current = drive_now + kick_current + sensory_current
        membrane_rhs_pa = (
            i_leak + exp_term - w + i_nap + i_rg_ltype_ca_pic + i_m + i_kca
            + i_rg_h + i_rg_t + i_rg_a + i_rg_gap
            + i_pf_slow + i_v2a_h + i_v2a_delay
            + i_v2a_gap_tonic + i_v2a_gap_phasic
            + i_v3_h + i_v3_t
            + i_renshaw_h + i_renshaw_sk
            + i_mn_coupling_soma + i_mn_ahp + direct_input_current + i_syn
            - mn_dendritic_synaptic_current
            + noise_multiplier * (eta + eta_pop[neuron_pop])
        )
        membrane_current_derivative_ns = (
            -leak + exp_term / cfg.slope_factor_mv
            + d_i_nap_dv + d_i_rg_ltype_ca_pic_dv + d_i_m_dv
            + d_i_kca_dv + d_i_rg_h_dv + d_i_rg_t_dv + d_i_rg_a_dv
            + d_i_rg_gap_dv + d_i_pf_slow_dv
            + d_i_v2a_h_tonic_dv + d_i_v2a_h_phasic_dv
            + d_i_v2a_h_delayed_dv + d_i_v2a_delay_dv
            + d_i_v2a_gap_tonic_dv + d_i_v2a_gap_phasic_dv
            + d_i_v3_h_ventral_dv + d_i_v3_h_dorsal_dv
            + d_i_v3_t_dorsal_dv + d_i_renshaw_h_dv
            + d_i_renshaw_sk_dv + d_i_mn_coupling_soma_dv
            + d_i_mn_ahp_dv + d_i_syn_dv
            - d_mn_dendritic_synaptic_current_dv
        )
        voltage_rhs_per_ms = membrane_rhs_pa / capacitance
        voltage_jacobian_per_ms = membrane_current_derivative_ns / capacitance
        dv, voltage_effective_jacobian_per_ms = (
            exponential_rosenbrock_increment(
                voltage_rhs_per_ms, voltage_jacobian_per_ms, dt
            )
        )
        adaptation_target = adaptation_a * (v - cfg.leak_reversal_mv)
        dw = (adaptation_target - w) * (
            1.0 - np.exp(-dt / adaptation_tau)
        )
        # Auditable class-level currents are sampled on the same physical
        # right-endpoint grid as the public trace, not inferred from metadata.
        if sample_idx < n_sample and step + 1 == sample_end_steps[sample_idx]:
            mechanism_abs_sum += np.asarray([
                float(np.mean(np.abs(
                    i_nap[is_rg] + i_rg_ltype_ca_pic[is_rg]
                    + i_m[is_rg] + i_kca[is_rg] + i_rg_h[is_rg]
                    + i_rg_t[is_rg] + i_rg_a[is_rg] + i_rg_gap[is_rg]
                ))),
                float(np.mean(np.abs(i_pf_slow[is_pf]))),
                float(np.mean(np.abs(
                    i_mn_nap_pic[is_mn] + i_mn_ltype_ca_pic[is_mn]
                    + i_mn_coupling_soma[is_mn]
                    + i_mn_ahp[is_mn]
                ))),
                float(np.mean(np.abs(
                    i_v2a_h[is_v2a] + i_v2a_delay[is_v2a]
                    + i_v2a_gap_tonic[is_v2a] + i_v2a_gap_phasic[is_v2a]
                ))),
                float(np.mean(np.abs(
                    i_v3_h[is_v3] + i_v3_t[is_v3]
                ))),
                float(np.mean(np.abs(
                    i_renshaw_nachr[is_renshaw]
                    + i_renshaw_glutamate[is_renshaw]
                    + i_renshaw_h[is_renshaw]
                    + i_renshaw_sk[is_renshaw]
                ))),
            ])
            renshaw_mixed_input_abs_sum += np.asarray([
                float(np.mean(np.abs(i_renshaw_nachr[is_renshaw]))),
                float(np.mean(np.abs(i_renshaw_glutamate[is_renshaw]))),
            ])
            mechanism_observation_count += 1
            runtime_intrinsic_current = {
                "I_LEAK": i_leak,
                "I_ADEX_EXP": exp_term,
                "I_ADAPT_W": -w,
                "I_NAP_RG": i_nap,
                "I_LTYPE_CA_PIC_RG": i_rg_ltype_ca_pic,
                "I_M_RG": i_m,
                "I_KCA_RG": i_kca,
                "I_H_RG": i_rg_h,
                "I_T_RG": i_rg_t,
                "I_A_RG": i_rg_a,
                "I_RG_GAP": i_rg_gap,
                "I_PF_SLOW": i_pf_slow,
                "I_MN_DEND_LEAK": i_mn_dendrite_leak,
                "I_MN_NAP_PIC": i_mn_nap_pic,
                "I_MN_LTYPE_CA_PIC": i_mn_ltype_ca_pic,
                "I_MN_COUPLING_SOMA": i_mn_coupling_soma,
                "I_MN_AHP": i_mn_ahp,
                "I_V2A_H_TONIC": i_v2a_h_tonic,
                "I_V2A_H_PHASIC": i_v2a_h_phasic,
                "I_V2A_H_DELAYED": i_v2a_h_delayed,
                "I_V2A_DELAY": i_v2a_delay,
                "I_V2A_GAP_TONIC": i_v2a_gap_tonic,
                "I_V2A_GAP_PHASIC": i_v2a_gap_phasic,
                "I_V3_H_VENTRAL": i_v3_h_ventral,
                "I_V3_H_DORSAL": i_v3_h_dorsal,
                "I_V3_T_DORSAL": i_v3_t_dorsal,
                "I_RENSHAW_H": i_renshaw_h,
                "I_RENSHAW_SK": i_renshaw_sk,
            }
            runtime_direct_input_current = {
                "I_TONIC_CLASS": tonic_class_current,
                "I_DESCENDING_RG": descending_rg_current,
                "I_PERTURBATION": perturbation_current,
                "I_IA_TO_PF_EFFECTIVE": ia_to_pf_current,
                "I_IA_TO_MN": ia_to_mn_current,
                "I_IA_TO_V1IA": ia_to_v1ia_current,
                "I_IB_TO_PF_EFFECTIVE": ib_to_pf_effective_current,
                "I_IB_TO_MN_EFFECTIVE": ib_to_mn_effective_current,
                "I_PF_DELETION": pf_deletion_current,
            }
            if set(runtime_intrinsic_current) != RUNTIME_INTRINSIC_TERM_IDS:
                raise RuntimeError("runtime intrinsic-current registry drift")
            if set(runtime_direct_input_current) != DIRECT_INPUT_TERM_IDS:
                raise RuntimeError("runtime direct-input current registry drift")
            reconstructed_direct_input = np.sum(np.column_stack([
                runtime_direct_input_current[term_id]
                for term_id in RUNTIME_DIRECT_INPUT_TERM_ORDER
            ]), axis=1)
            if not np.allclose(
                reconstructed_direct_input, direct_input_current,
                rtol=0.0, atol=1e-10,
            ):
                raise RuntimeError("direct-input term accounting drift")
            class_membrane_rhs_abs_sum += np.bincount(
                neuron_class_index, weights=np.abs(membrane_rhs_pa),
                minlength=len(CLASSES),
            ) / class_neuron_count
            class_synaptic_current_abs_sum += np.bincount(
                neuron_class_index, weights=np.abs(i_syn),
                minlength=len(CLASSES),
            ) / class_neuron_count
            class_direct_input_current_abs_sum += np.bincount(
                neuron_class_index, weights=np.abs(direct_input_current),
                minlength=len(CLASSES),
            ) / class_neuron_count
            runtime_current_matrix = np.column_stack([
                runtime_intrinsic_current[term_id]
                for term_id in RUNTIME_INTRINSIC_TERM_ORDER
            ])
            class_term_mean_abs = np.add.reduceat(
                np.abs(runtime_current_matrix), class_neuron_start, axis=0
            ) / class_neuron_count[:, None]
            class_intrinsic_term_abs_sum += np.where(
                class_intrinsic_term_declared, class_term_mean_abs, 0.0
            )
            runtime_direct_input_matrix = np.column_stack([
                runtime_direct_input_current[term_id]
                for term_id in RUNTIME_DIRECT_INPUT_TERM_ORDER
            ])
            class_direct_term_mean_abs = np.add.reduceat(
                np.abs(runtime_direct_input_matrix),
                class_neuron_start,
                axis=0,
            ) / class_neuron_count[:, None]
            class_direct_input_term_abs_sum += np.where(
                class_direct_input_term_declared,
                class_direct_term_mean_abs,
                0.0,
            )
            class_current_observation_count += 1

        v_before_update = v.copy()
        refractory_before_update = refractory.copy()
        active = refractory_before_update <= 0.0
        # A neuron whose refractory interval ends inside this integration
        # step resumes from reset for the physical remainder of the step.
        # The v2.5 right-endpoint reduction held it at reset for the whole
        # interval, introducing a dt-dependent extra refractory delay.
        releasing = (
            (refractory_before_update > 0.0)
            & (refractory_before_update < dt)
        )
        active_interval_ms = np.where(
            active,
            dt,
            np.where(releasing, dt - refractory_before_update, 0.0),
        )
        active_start_fraction = np.where(
            releasing, refractory_before_update / dt, 0.0
        )
        integration_initial_voltage = v_before_update.copy()
        integration_initial_voltage[releasing] = cfg.reset_mv
        step_effective_jacobian_per_ms = (
            voltage_effective_jacobian_per_ms.copy()
        )
        active_mn_dendrite = is_mn & ~ablated_neuron
        mn_dendrite_v[active_mn_dendrite] += d_mn_dendrite_v[
            active_mn_dendrite
        ]
        v[active] += dv[active]
        v[~active] = cfg.reset_mv
        if np.any(releasing):
            releasing_dv, releasing_effective_jacobian = (
                exponential_rosenbrock_increment(
                    voltage_rhs_per_ms[releasing],
                    voltage_jacobian_per_ms[releasing],
                    active_interval_ms[releasing],
                )
            )
            v[releasing] = (
                cfg.reset_mv + releasing_dv
            )
            step_effective_jacobian_per_ms[releasing] = (
                releasing_effective_jacobian
            )
        w += dw
        refractory = np.maximum(0.0, refractory_before_update - dt)

        spikes = (v >= cfg.spike_peak_mv) & (refractory <= 0.0)
        spikes &= ~ablated_neuron
        rate_event_increment = np.zeros(n_pop)
        target_event_fractions = np.asarray([], dtype=float)
        if np.any(spikes):
            spike_ids = np.where(spikes)[0]
            spike_pops = neuron_pop[spike_ids]
            crossing_fraction = locally_linearized_threshold_fraction(
                integration_initial_voltage[spike_ids],
                cfg.spike_peak_mv,
                voltage_rhs_per_ms[spike_ids],
                step_effective_jacobian_per_ms[spike_ids],
                active_interval_ms[spike_ids],
                active_start_fraction[spike_ids],
                dt,
                v[spike_ids],
            )
            target_event_fractions = crossing_fraction[
                spike_pops == pulse_population
            ]
            spike_times = t_s + crossing_fraction * dt / 1000.0
            spike_time.extend(spike_times.tolist())
            spike_population.extend(spike_pops.tolist())
            spike_neuron.extend(neuron_local[spike_ids].tolist())
            counts = np.bincount(spike_pops, minlength=n_pop).astype(float)
            class_spike_count += np.bincount(
                neuron_class_index[spike_ids], minlength=len(CLASSES)
            ).astype(np.int64)
            np.add.at(
                rate_event_increment,
                spike_pops,
                1000.0
                / (sizes[spike_pops] * cfg.rate_tau_ms)
                * right_endpoint_event_decay(
                    crossing_fraction, dt, cfg.rate_tau_ms
                ),
            )
            if fast_mode == "dynamic":
                calcium[spike_ids] += (
                    cfg.calcium_spike_increment
                    * rg_kca_positive_mask[spike_ids]
                    * right_endpoint_event_decay(
                        crossing_fraction, dt, cfg.calcium_decay_ms
                    )
                )
            mn_calcium[spike_ids] += (
                cfg.mn_calcium_spike_increment * is_mn[spike_ids]
                * right_endpoint_event_decay(
                    crossing_fraction, dt, cfg.mn_calcium_decay_ms
                )
            )
            renshaw_calcium[spike_ids] += (
                cfg.renshaw_calcium_spike_increment * is_renshaw[spike_ids]
                * right_endpoint_event_decay(
                    crossing_fraction, dt, cfg.renshaw_calcium_decay_ms
                )
            )

            for source_id, source_crossing_fraction in zip(
                spike_ids, crossing_fraction
            ):
                source_population_id = int(neuron_pop[source_id])
                nmj_context = int(mn_population_context[source_population_id])
                if nmj_context >= 0:
                    event_fraction = 1.0 / float(sizes[source_population_id])
                    terminal_release_event_time.append(
                        t_s
                        + float(source_crossing_fraction) * dt / 1000.0
                        + cfg.nmj_release_delay_ms / 1000.0
                    )
                    terminal_release_event_context.append(nmj_context)
                    terminal_release_event_fraction.append(event_fraction)
                    (
                        nmj_lower,
                        nmj_upper,
                        nmj_lower_weight,
                        nmj_upper_weight,
                    ) = fractional_delay_bins(
                        float(source_crossing_fraction),
                        cfg.nmj_release_delay_ms,
                        dt,
                    )
                    nmj_event_split_mass_max_abs_error = max(
                        nmj_event_split_mass_max_abs_error,
                        abs(float(nmj_lower_weight + nmj_upper_weight) - 1.0),
                    )
                    reconstructed_nmj_delay_ms = (
                        float(nmj_lower) * float(nmj_lower_weight)
                        + float(nmj_upper) * float(nmj_upper_weight)
                        - float(source_crossing_fraction)
                    ) * dt
                    nmj_delay_reconstruction_max_abs_error_ms = max(
                        nmj_delay_reconstruction_max_abs_error_ms,
                        abs(reconstructed_nmj_delay_ms - cfg.nmj_release_delay_ms),
                    )
                    scheduled_nmj_event_fraction[
                        (nmj_ring_pointer + int(nmj_lower)) % nmj_ring_length,
                        nmj_context,
                    ] += event_fraction * float(nmj_lower_weight)
                    scheduled_nmj_event_fraction[
                        (nmj_ring_pointer + int(nmj_upper)) % nmj_ring_length,
                        nmj_context,
                    ] += event_fraction * float(nmj_upper_weight)
                    nmj_scheduled_event_fraction_total += event_fraction

                edge_ids = connectome["outgoing"][int(source_id)]
                if not len(edge_ids):
                    continue
                active_v3_motor_edges = (
                    v3_motor_edge_mask[edge_ids]
                    & ~ablated_edge_mask[edge_ids]
                )
                if np.any(active_v3_motor_edges):
                    subtype_index = int(v3_dorsal_mask[int(source_id)])
                    v3_motor_scheduled_edge_event_count_by_subphenotype[
                        subtype_index
                    ] += int(np.sum(active_v3_motor_edges))
                active_v3_microcircuit_edges = (
                    v3_microcircuit_edge_masks[:, edge_ids]
                    & ~ablated_edge_mask[edge_ids][None, :]
                )
                v3_microcircuit_scheduled_edge_event_counts += np.sum(
                    active_v3_microcircuit_edges, axis=1, dtype=np.int64
                )
                amplitudes = connectome["weight_pa"][edge_ids].copy()
                amplitudes[ablated_edge_mask[edge_ids]] = 0.0
                sensitive = resource_edge_mask[edge_ids] & ~ablated_edge_mask[edge_ids]
                if np.any(sensitive):
                    sensitive_edges = edge_ids[sensitive]
                    aged_activity_increment = (
                        cfg.mt_activity_spike_increment
                        * float(right_endpoint_event_decay(
                            float(source_crossing_fraction),
                            dt,
                            cfg.mt_activity_tau_ms,
                        ))
                    )
                    if mt_mode in {"dynamic", "impaired"}:
                        mt_edge_activity[sensitive_edges] += aged_activity_increment
                    elif mt_mode == "spatial_shuffled":
                        mt_edge_activity[
                            spatial_edge_map[sensitive_edges]
                        ] += aged_activity_increment
                    amplitudes[sensitive] *= edge_rrp_available[sensitive_edges]
                    demand_depletion = 1.35 if demand_active else 1.0
                    edge_rrp_available[sensitive_edges] *= np.maximum(
                        0.0,
                        1.0 - demand_depletion * depletion_fraction[sensitive_edges],
                    )
                (
                    lower_offsets,
                    upper_offsets,
                    lower_weights,
                    upper_weights,
                ) = fractional_delay_bins(
                    float(source_crossing_fraction),
                    connectome["delay_ms"][edge_ids],
                    dt,
                )
                central_event_split_mass_max_abs_error = max(
                    central_event_split_mass_max_abs_error,
                    float(np.max(np.abs(lower_weights + upper_weights - 1.0))),
                )
                reconstructed_delays_ms = (
                    lower_offsets * lower_weights
                    + upper_offsets * upper_weights
                    - float(source_crossing_fraction)
                ) * dt
                central_delay_reconstruction_max_abs_error_ms = max(
                    central_delay_reconstruction_max_abs_error_ms,
                    float(np.max(np.abs(
                        reconstructed_delays_ms
                        - connectome["delay_ms"][edge_ids]
                    ))),
                )
                lower_slots = (ring_pointer + lower_offsets) % ring_length
                upper_slots = (ring_pointer + upper_offsets) % ring_length
                nachr = renshaw_nachr_edge_mask[edge_ids]
                renshaw_glutamate = renshaw_glutamate_edge_mask[edge_ids]
                special_renshaw = nachr | renshaw_glutamate
                excitatory = (amplitudes > 0.0) & ~special_renshaw
                inhibitory = (amplitudes < 0.0) & ~special_renshaw
                if np.any(excitatory):
                    conductance = amplitudes[excitatory] / (
                        cfg.excitatory_reversal_mv
                        - cfg.synaptic_reference_voltage_mv
                    )
                    np.add.at(
                        scheduled_exc_conductance,
                        (
                            lower_slots[excitatory],
                            connectome["target"][edge_ids[excitatory]],
                        ),
                        conductance * lower_weights[excitatory],
                    )
                    np.add.at(
                        scheduled_exc_conductance,
                        (
                            upper_slots[excitatory],
                            connectome["target"][edge_ids[excitatory]],
                        ),
                        conductance * upper_weights[excitatory],
                    )
                if np.any(inhibitory):
                    conductance = -amplitudes[inhibitory] / (
                        cfg.synaptic_reference_voltage_mv
                        - cfg.inhibitory_reversal_mv
                    )
                    np.add.at(
                        scheduled_inh_conductance,
                        (
                            lower_slots[inhibitory],
                            connectome["target"][edge_ids[inhibitory]],
                        ),
                        conductance * lower_weights[inhibitory],
                    )
                    np.add.at(
                        scheduled_inh_conductance,
                        (
                            upper_slots[inhibitory],
                            connectome["target"][edge_ids[inhibitory]],
                        ),
                        conductance * upper_weights[inhibitory],
                    )
                if np.any(nachr):
                    conductance = np.maximum(
                        amplitudes[nachr], 0.0
                    ) / (
                        cfg.excitatory_reversal_mv
                        - cfg.synaptic_reference_voltage_mv
                    )
                    np.add.at(
                        scheduled_nachr_conductance,
                        (
                            lower_slots[nachr],
                            connectome["target"][edge_ids[nachr]],
                        ),
                        conductance * lower_weights[nachr],
                    )
                    np.add.at(
                        scheduled_nachr_conductance,
                        (
                            upper_slots[nachr],
                            connectome["target"][edge_ids[nachr]],
                        ),
                        conductance * upper_weights[nachr],
                    )
                if np.any(renshaw_glutamate):
                    conductance = np.maximum(
                        amplitudes[renshaw_glutamate], 0.0
                    ) / (
                        cfg.excitatory_reversal_mv
                        - cfg.synaptic_reference_voltage_mv
                    )
                    np.add.at(
                        scheduled_renshaw_glutamate_conductance,
                        (
                            lower_slots[renshaw_glutamate],
                            connectome["target"][edge_ids[renshaw_glutamate]],
                        ),
                        conductance * lower_weights[renshaw_glutamate],
                    )
                    np.add.at(
                        scheduled_renshaw_glutamate_conductance,
                        (
                            upper_slots[renshaw_glutamate],
                            connectome["target"][edge_ids[renshaw_glutamate]],
                        ),
                        conductance * upper_weights[renshaw_glutamate],
                    )
            v[spikes] = cfg.reset_mv
            # The spike-triggered adaptation jump is placed at the interpolated
            # event time and aged over the within-step remainder.  Refractory
            # time is likewise measured from that event rather than from the
            # common right endpoint.  Reset voltage remains clamped for the
            # remainder, as required by the represented refractory reduction.
            spike_remainder_ms = (1.0 - crossing_fraction) * dt
            w[spikes] += adaptation_b[spikes] * np.exp(
                -spike_remainder_ms / adaptation_tau[spikes]
            )
            refractory[spikes] = np.maximum(
                0.0, cfg.refractory_ms - spike_remainder_ms
            )
        else:
            counts = np.zeros(n_pop)

        # Preferred replay API: physical events are included at the first common
        # right endpoint at or after their timestamp, then aged only over the
        # within-step remainder. This matches dynamic spike-triggered jumps.
        if fast_mode == "yoked" and kca_time_api:
            kca_event_end = int(np.searchsorted(
                kca_event_times_s,
                np.nextafter(step_end_s, math.inf),
                side="right",
            ))
            if kca_event_end > kca_event_cursor:
                event_slice = slice(kca_event_cursor, kca_event_end)
                event_times_s = kca_event_times_s[event_slice]
                event_neurons = kca_event_neurons[event_slice]
                event_fractions = np.clip(
                    (event_times_s - t_s) * 1000.0 / dt, 0.0, 1.0
                )
                np.add.at(
                    calcium,
                    event_neurons,
                    cfg.calcium_spike_increment
                    * right_endpoint_event_decay(
                        event_fractions, dt, cfg.calcium_decay_ms
                    ),
                )
            kca_event_cursor = kca_event_end
        if mt_mode == "time_yoked" and mt_time_api:
            mt_event_end = int(np.searchsorted(
                mt_event_times_s,
                np.nextafter(step_end_s, math.inf),
                side="right",
            ))
            if mt_event_end > mt_event_cursor:
                event_slice = slice(mt_event_cursor, mt_event_end)
                event_times_s = mt_event_times_s[event_slice]
                event_edges = mt_event_edges[event_slice]
                event_fractions = np.clip(
                    (event_times_s - t_s) * 1000.0 / dt, 0.0, 1.0
                )
                np.add.at(
                    mt_edge_activity,
                    event_edges,
                    cfg.mt_activity_spike_increment
                    * right_endpoint_event_decay(
                        event_fractions, dt, cfg.mt_activity_tau_ms
                    ),
                )
            mt_event_cursor = mt_event_end

        rate = rate * rate_decay + rate_event_increment
        rate[ablated_population_mask] = 0.0
        v[ablated_neuron] = cfg.reset_mv
        mn_dendrite_v[ablated_neuron & is_mn] = cfg.reset_mv
        w[ablated_neuron] = 0.0
        rg_rates = rate[rg_context_indices]
        mn_rates = rate[mn_context_indices]
        target_rate = float(rate[pulse_population])
        (
            reconstructed_target_rate,
            target_burst_armed,
            target_onset_fractions,
        ) = advance_exponential_rate_hysteresis(
            previous_target_rate,
            target_event_fractions,
            1000.0 / (sizes[pulse_population] * cfg.rate_tau_ms),
            dt,
            cfg.rate_tau_ms,
            cfg.burst_on_threshold_hz,
            cfg.burst_off_threshold_hz,
            target_burst_armed,
        )
        if not math.isclose(
            reconstructed_target_rate,
            target_rate,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("online target-rate reconstruction drifted")
        for onset_fraction in target_onset_fractions:
            onset_time_s = t_s + float(onset_fraction) * dt / 1000.0
            target_online_onsets.append(onset_time_s)
            if (
                protocol == "pulse"
                and sham_excitatory_start_step < 0
                and onset_time_s >= cfg.pulse_arm_after_s
                and len(target_online_onsets) >= 2
            ):
                period_s = target_online_onsets[-1] - target_online_onsets[-2]
                excitatory_fraction = (
                    pulse_cycle_fraction_override
                    if pulse_direction == "excitatory"
                    and pulse_cycle_fraction_override is not None
                    else cfg.excitatory_pulse_cycle_fraction
                )
                inhibitory_fraction = (
                    pulse_cycle_fraction_override
                    if pulse_direction == "inhibitory"
                    and pulse_cycle_fraction_override is not None
                    else cfg.inhibitory_pulse_cycle_fraction
                )
                sham_excitatory_start_step = int(math.ceil(
                    (onset_time_s + excitatory_fraction * period_s)
                    * 1000.0 / dt - 1.0e-12
                ))
                sham_inhibitory_start_step = int(math.ceil(
                    (onset_time_s + inhibitory_fraction * period_s)
                    * 1000.0 / dt - 1.0e-12
                ))
                pulse_steps = int(round(cfg.pulse_duration_ms / dt))
                sham_excitatory_end_step = (
                    sham_excitatory_start_step + pulse_steps
                )
                sham_inhibitory_end_step = (
                    sham_inhibitory_start_step + pulse_steps
                )
                if pulse_direction == "excitatory":
                    pulse_start_step = sham_excitatory_start_step
                    pulse_end_step = sham_excitatory_end_step
                elif pulse_direction == "inhibitory":
                    pulse_start_step = sham_inhibitory_start_step
                    pulse_end_step = sham_inhibitory_end_step
                if pulse_start_step >= 0:
                    pulse_trigger_time_s = pulse_start_step * dt / 1000.0
        previous_target_rate = target_rate

        if sample_idx < n_sample and step + 1 == sample_end_steps[sample_idx]:
            sample_time[sample_idx] = step_end_s
            sample_rate[sample_idx] = rate
            # Route and population aggregates are readouts only.  Computing
            # them on the declared 1-ms public grid preserves their physical
            # sampling contract without re-averaging unchanged edge groupings
            # at every sub-millisecond integration step.
            mt_route_support = np.asarray([
                float(np.mean(edge_effective_tracks[mask]))
                if np.any(mask) else 0.0
                for mask in route_edge_masks
            ])
            mt_route_activity = np.asarray([
                float(np.mean(mt_edge_activity[mask]))
                if np.any(mask) else 0.0
                for mask in route_edge_masks
            ])
            mt_mean_pop = np.zeros(n_pop)
            routed = population_route_index >= 0
            mt_mean_pop[routed] = mt_route_support[
                population_route_index[routed]
            ]
            sample_mt_pop[sample_idx] = mt_mean_pop
            for side_id in range(len(SIDES)):
                side_edge_mask = resource_edge_mask & (edge_side_index == side_id)
                side_support = edge_effective_tracks[side_edge_mask]
                side_activity = mt_edge_activity[side_edge_mask]
                sample_mt_side[sample_idx, side_id] = float(np.mean(side_support))
                sample_mt_activity_side[sample_idx, side_id] = float(
                    np.mean(side_activity)
                )
            sample_mt_route[sample_idx] = mt_route_support
            sample_mt_route_activity[sample_idx] = mt_route_activity
            sample_calcium_side[sample_idx] = [
                float(np.mean(calcium[mask & is_rg])) for mask in mt_side_masks
            ]
            endpoint_calcium_hill = np.power(
                np.maximum(calcium, 0.0), cfg.calcium_hill_coefficient
            )
            endpoint_kca_activation = endpoint_calcium_hill / (
                endpoint_calcium_hill
                + cfg.calcium_half_activation ** cfg.calcium_hill_coefficient
            )
            if fast_mode == "off":
                endpoint_kca_activation.fill(0.0)
            elif fast_mode == "static_mean":
                endpoint_kca_activation[is_rg] = cfg.static_kca_activation_reference
            elif fast_mode == "yoked":
                endpoint_kca_activation *= fast_activation_scale
            endpoint_kca_activation *= rg_kca_positive_mask
            sample_kca_activation_side[sample_idx] = [
                float(np.mean(endpoint_kca_activation[mask & is_rg]))
                for mask in mt_side_masks
            ]
            sample_rrp_route[sample_idx] = [
                float(np.mean(edge_rrp_available[mask])) if np.any(mask) else 1.0
                for mask in route_edge_masks
            ]
            sample_replenishment_resource_route[sample_idx] = [
                float(np.mean(edge_slow_replenishment_resource[mask]))
                if np.any(mask) else 1.0
                for mask in route_edge_masks
            ]
            sample_ia_signal[sample_idx] = latest_ia_signal
            sample_ib_signal[sample_idx] = latest_ib_signal
            sample_ia_transmission[sample_idx] = latest_ia_transmission
            sample_ib_transmission[sample_idx] = latest_ib_transmission
            sample_nmj_vesicle[sample_idx] = nmj_vesicle_available
            sample_nmj_ach_gate[sample_idx] = nmj_ach_gate
            sample_nmj_endplate_mv[sample_idx] = nmj_endplate_mv
            sample_nmj_nachr_open[sample_idx] = nmj_nachr_open
            sample_muscle_fiber_excitation[sample_idx] = muscle_fiber_excitation
            sample_muscle_calcium[sample_idx] = muscle_calcium
            sample_muscle_activation[sample_idx] = muscle_activation
            sample_muscle_force[sample_idx] = muscle_force
            sample_joint_state[sample_idx] = np.r_[joint_position, joint_velocity]
            sample_mn[sample_idx] = mn_rates
            sample_challenged_rrp[sample_idx] = (
                float(np.mean(edge_rrp_available[challenged_edges]))
                if np.any(challenged_edges) else 1.0
            )
            sample_challenged_replenishment_resource[sample_idx] = (
                float(np.mean(
                    edge_slow_replenishment_resource[challenged_edges]
                ))
                if np.any(challenged_edges) else 1.0
            )
            sample_idx += 1
        ring_pointer = (ring_pointer + 1) % ring_length
        nmj_ring_pointer = (nmj_ring_pointer + 1) % nmj_ring_length

    return {
        "time_s": sample_time[:sample_idx],
        "rate_hz": sample_rate[:sample_idx],
        "population_names": np.asarray(POPULATIONS),
        "population_equation_id": np.asarray([
            CELL_CLASS_EQUATIONS[name.rsplit("_", 2)[0]].equation_id
            for name in POPULATIONS
        ]),
        "class_execution_contract_name": np.asarray(CLASSES),
        "class_execution_contract_id": np.asarray([
            CLASS_EXECUTION_CONTRACTS[cell_class].contract_id
            for cell_class in CLASSES
        ]),
        "class_intrinsic_runtime_term_ids": np.asarray([
            "|".join(CLASS_EXECUTION_CONTRACTS[cell_class].intrinsic_term_ids)
            for cell_class in CLASSES
        ]),
        "runtime_intrinsic_term_names": np.asarray(
            RUNTIME_INTRINSIC_TERM_ORDER
        ),
        "class_intrinsic_term_declared_mask": class_intrinsic_term_declared,
        "class_intrinsic_term_mean_abs_pa": (
            class_intrinsic_term_abs_sum
            / max(class_current_observation_count, 1)
        ),
        "class_direct_input_term_ids": np.asarray([
            "|".join(CLASS_EXECUTION_CONTRACTS[cell_class].direct_input_ids)
            for cell_class in CLASSES
        ]),
        "runtime_direct_input_term_names": np.asarray(
            RUNTIME_DIRECT_INPUT_TERM_ORDER
        ),
        "class_direct_input_term_declared_mask": (
            class_direct_input_term_declared
        ),
        "class_direct_input_term_mean_abs_pa": (
            class_direct_input_term_abs_sum
            / max(class_current_observation_count, 1)
        ),
        "class_peripheral_output_term_ids": np.asarray([
            "|".join(CLASS_EXECUTION_CONTRACTS[cell_class].peripheral_output_ids)
            for cell_class in CLASSES
        ]),
        "class_membrane_rhs_mean_abs_pa": (
            class_membrane_rhs_abs_sum
            / max(class_current_observation_count, 1)
        ),
        "class_synaptic_current_mean_abs_pa": (
            class_synaptic_current_abs_sum
            / max(class_current_observation_count, 1)
        ),
        "class_direct_input_current_mean_abs_pa": (
            class_direct_input_current_abs_sum
            / max(class_current_observation_count, 1)
        ),
        "class_spike_count": class_spike_count,
        "class_incoming_pathway_count": np.asarray([
            len({
                spec.name for spec in connectome["pathways"]
                if POPULATIONS[spec.target_population].rsplit("_", 2)[0]
                == cell_class
            })
            for cell_class in CLASSES
        ], dtype=np.int16),
        "class_outgoing_pathway_count": np.asarray([
            len({
                spec.name for spec in connectome["pathways"]
                if POPULATIONS[spec.source_population].rsplit("_", 2)[0]
                == cell_class
            })
            for cell_class in CLASSES
        ], dtype=np.int16),
        "class_incoming_connectome_edge_count": np.bincount(
            neuron_class_index[connectome["target"]], minlength=len(CLASSES)
        ).astype(np.int64),
        "class_outgoing_connectome_edge_count": np.bincount(
            neuron_class_index[connectome["source"]], minlength=len(CLASSES)
        ).astype(np.int64),
        "class_outgoing_ablated_edge_count": np.bincount(
            neuron_class_index[connectome["source"]],
            weights=ablated_edge_mask.astype(np.int64),
            minlength=len(CLASSES),
        ).astype(np.int64),
        "biological_interface_names": np.asarray(tuple(BIOLOGICAL_INTERFACE_EQUATIONS)),
        "biological_interface_equation_id": np.asarray([
            record.equation_id for record in BIOLOGICAL_INTERFACE_EQUATIONS.values()
        ]),
        "mt_population": sample_mt_pop[:sample_idx],
        "mt_side": sample_mt_side[:sample_idx],
        "mt_activity_side": sample_mt_activity_side[:sample_idx],
        "mt_route_support": sample_mt_route[:sample_idx],
        "mt_route_activity": sample_mt_route_activity[:sample_idx],
        "mt_route_names": np.asarray(MT_ROUTES),
        "calcium_side": sample_calcium_side[:sample_idx],
        "kca_activation_side": sample_kca_activation_side[:sample_idx],
        "rrp_route_mean": sample_rrp_route[:sample_idx],
        "replenishment_resource_route_mean": (
            sample_replenishment_resource_route[:sample_idx]
        ),
        "vesicle_route_names": np.asarray(MT_ROUTES),
        "ia_signal": sample_ia_signal[:sample_idx],
        "ib_signal": sample_ib_signal[:sample_idx],
        "ia_transmission": sample_ia_transmission[:sample_idx],
        "ib_transmission": sample_ib_transmission[:sample_idx],
        "nmj_vesicle_available": sample_nmj_vesicle[:sample_idx],
        "nmj_ach_gate": sample_nmj_ach_gate[:sample_idx],
        "nmj_endplate_mv": sample_nmj_endplate_mv[:sample_idx],
        "nmj_nachr_open": sample_nmj_nachr_open[:sample_idx],
        "muscle_fiber_excitation": sample_muscle_fiber_excitation[:sample_idx],
        "muscle_calcium": sample_muscle_calcium[:sample_idx],
        "muscle_activation": sample_muscle_activation[:sample_idx],
        "muscle_force": sample_muscle_force[:sample_idx],
        "joint_state": sample_joint_state[:sample_idx],
        "mn_hz": sample_mn[:sample_idx],
        "challenged_rrp": sample_challenged_rrp[:sample_idx],
        "challenged_replenishment_resource": (
            sample_challenged_replenishment_resource[:sample_idx]
        ),
        "challenged_edges": challenged_edges,
        "spike_time_s": np.asarray(spike_time),
        "neuron_event_time_s": np.asarray(spike_time),
        "spike_population": np.asarray(spike_population, dtype=np.int16),
        "spike_neuron": np.asarray(spike_neuron, dtype=np.int16),
        "terminal_release_event_time_s": np.asarray(terminal_release_event_time),
        "terminal_release_event_context": np.asarray(
            terminal_release_event_context, dtype=np.int8
        ),
        "terminal_release_event_fraction": np.asarray(
            terminal_release_event_fraction, dtype=float
        ),
        "terminal_release_delivery_time_s": np.asarray(
            terminal_release_delivery_time
        ),
        "terminal_release_delivery_context": np.asarray(
            terminal_release_delivery_context, dtype=np.int8
        ),
        "terminal_release_delivery_fraction": np.asarray(
            terminal_release_delivery_fraction, dtype=float
        ),
        "nmj_scheduled_event_fraction_total": np.asarray([
            nmj_scheduled_event_fraction_total
        ]),
        "nmj_arrived_event_fraction_total": np.asarray([
            nmj_arrived_event_fraction_total
        ]),
        "nmj_pending_event_fraction_total": np.asarray([
            float(np.sum(scheduled_nmj_event_fraction))
        ]),
        "nmj_delay_reconstruction_max_abs_error_ms": np.asarray([
            nmj_delay_reconstruction_max_abs_error_ms
        ]),
        "nmj_event_split_mass_max_abs_error": np.asarray([
            nmj_event_split_mass_max_abs_error
        ]),
        "central_delay_reconstruction_max_abs_error_ms": np.asarray([
            central_delay_reconstruction_max_abs_error_ms
        ]),
        "central_event_split_mass_max_abs_error": np.asarray([
            central_event_split_mass_max_abs_error
        ]),
        "mt_eligible_population": mt_eligible_pop,
        "population_sizes": sizes,
        "connectome_source": connectome["source"],
        "connectome_target": connectome["target"],
        "connectome_weight_pa": connectome["weight_pa"],
        "connectome_delay_ms": connectome["delay_ms"],
        "connectome_functional_role": connectome["functional_role"],
        "connectome_mt_route": connectome["mt_route"],
        "connectome_recruitment_axis": connectome["recruitment_axis"],
        "connectome_evidence_class": connectome["evidence_class"],
        "connectome_evidence_note": connectome["evidence_note"],
        "connectome_transmitter_identity": connectome[
            "transmitter_identity"
        ],
        "connectome_topology_group": connectome["topology_group"],
        "connectome_source_subphenotype_selector": connectome[
            "source_subphenotype"
        ],
        "connectome_target_subphenotype_selector": connectome[
            "target_subphenotype"
        ],
        "connectome_pathway_index": connectome["pathway_index"],
        "connectome_ablated_edge_mask": ablated_edge_mask,
        "edge_mt_activity_final": mt_edge_activity,
        "edge_mt_track_final": edge_effective_tracks,
        "edge_rrp_final": edge_rrp_available,
        "edge_slow_replenishment_resource_final": (
            edge_slow_replenishment_resource
        ),
        "sensory_resource_final": sensory_resource,
        "v2a_variant": v2a_variant,
        "v2a_h_positive_mask": v2a_h_positive_mask,
        "v2a_h_phenotype_names": np.asarray(["tonic", "phasic", "delayed"]),
        "v2a_h_positive_counts_by_phenotype": np.asarray([
            np.sum(v2a_h_positive_mask & (v2a_variant == variant_id))
            for variant_id in range(3)
        ], dtype=np.int64),
        "v2a_h_total_counts_by_phenotype": np.asarray([
            np.sum(is_v2a & (v2a_variant == variant_id))
            for variant_id in range(3)
        ], dtype=np.int64),
        "v2a_h_source_positive_fractions": np.asarray(
            cfg.v2a_h_positive_fractions
        ),
        "v2a_h_gate_final": v2a_h_gate,
        "v3_dorsal_mask": v3_dorsal_mask,
        "v3_vlat_connectivity_mask": v3_vlat_connectivity_mask,
        "v3_connectivity_subphenotype_names": np.asarray([
            "ventral_nonVLat_unresolved", "V3_VLat", "dorsal",
        ]),
        "v3_connectivity_subphenotype_cell_counts": np.asarray([
            np.sum(is_v3 & ~v3_dorsal_mask & ~v3_vlat_connectivity_mask),
            np.sum(v3_vlat_connectivity_mask),
            np.sum(v3_dorsal_mask),
        ], dtype=np.int64),
        "v3_microcircuit_pathway_names": v3_microcircuit_pathway_names,
        "v3_microcircuit_edge_counts": np.sum(
            v3_microcircuit_edge_masks, axis=1, dtype=np.int64
        ),
        "v3_microcircuit_scheduled_edge_event_counts": (
            v3_microcircuit_scheduled_edge_event_counts
        ),
        "v3_motor_source_subphenotype_names": np.asarray([
            "ventral", "dorsal",
        ]),
        "v3_motor_edge_count_by_source_subphenotype": np.asarray([
            np.sum(
                v3_motor_edge_mask
                & ~v3_dorsal_mask[connectome["source"]]
            ),
            np.sum(
                v3_motor_edge_mask
                & v3_dorsal_mask[connectome["source"]]
            ),
        ], dtype=np.int64),
        "v3_motor_scheduled_edge_event_count_by_source_subphenotype": (
            v3_motor_scheduled_edge_event_count_by_subphenotype
        ),
        "rg_pic_positive_mask": rg_pic_positive_mask,
        "rg_m_positive_mask": rg_m_positive_mask,
        "rg_kca_positive_mask": rg_kca_positive_mask,
        "rg_h_positive_mask": rg_h_positive_mask,
        "rg_t_positive_mask": rg_t_positive_mask,
        "rg_a_positive_mask": rg_a_positive_mask,
        "rg_current_subset_names": np.asarray([
            "PIC_shared_NaP_LCa", "M", "KCa_sAHP", "Ih", "IT", "IA",
        ]),
        "rg_current_subset_positive_counts": np.asarray([
            np.sum(rg_pic_positive_mask), np.sum(rg_m_positive_mask),
            np.sum(rg_kca_positive_mask), np.sum(rg_h_positive_mask),
            np.sum(rg_t_positive_mask), np.sum(rg_a_positive_mask),
        ], dtype=np.int64),
        "rg_current_subset_realized_fractions": np.asarray([
            np.mean(rg_pic_positive_mask[is_rg]),
            np.mean(rg_m_positive_mask[is_rg]),
            np.mean(rg_kca_positive_mask[is_rg]),
            np.mean(rg_h_positive_mask[is_rg]),
            np.mean(rg_t_positive_mask[is_rg]),
            np.mean(rg_a_positive_mask[is_rg]),
        ]),
        "rg_current_subset_source_fractions": np.asarray([
            cfg.rg_pic_positive_fraction, cfg.rg_m_positive_fraction,
            cfg.rg_kca_positive_fraction, cfg.rg_h_positive_fraction,
            cfg.rg_t_positive_fraction, cfg.rg_a_positive_fraction,
        ]),
        "rg_gap_edge_source": rg_gap_source,
        "rg_gap_edge_target": rg_gap_target,
        "v2a_tonic_gap_edge_source": v2a_tonic_gap_source,
        "v2a_tonic_gap_edge_target": v2a_tonic_gap_target,
        "v2a_phasic_gap_edge_source": v2a_phasic_gap_source,
        "v2a_phasic_gap_edge_target": v2a_phasic_gap_target,
        "gap_pair_graph_names": np.asarray([
            "RG_local", "V2a_tonic_local", "V2a_phasic_local",
        ]),
        "gap_pair_candidate_counts": np.asarray([
            rg_gap_candidate_pair_count,
            v2a_tonic_gap_candidate_pair_count,
            v2a_phasic_gap_candidate_pair_count,
        ], dtype=np.int64),
        "gap_pair_edge_counts": np.asarray([
            len(rg_gap_source), len(v2a_tonic_gap_source),
            len(v2a_phasic_gap_source),
        ], dtype=np.int64),
        "gap_pair_realized_probabilities": np.asarray([
            len(rg_gap_source) / max(rg_gap_candidate_pair_count, 1),
            len(v2a_tonic_gap_source) / max(
                v2a_tonic_gap_candidate_pair_count, 1
            ),
            len(v2a_phasic_gap_source) / max(
                v2a_phasic_gap_candidate_pair_count, 1
            ),
        ]),
        "rg_ltype_ca_pic_activation_final": rg_ltype_ca_pic_activation[is_rg],
        "rg_h_gate_final": rg_h_gate[is_rg],
        "rg_t_inactivation_final": rg_t_inactivation[is_rg],
        "rg_a_inactivation_final": rg_a_inactivation[is_rg],
        "membrane_voltage_final_mv": v,
        "drive_heterogeneity_scale": drive_scale,
        "synaptic_heterogeneity_scale": syn_scale,
        "pf_slow_gate_final": pf_slow_gate,
        "syn_exc_conductance_final_ns": syn_exc_conductance,
        "syn_inh_conductance_final_ns": syn_inh_conductance,
        "renshaw_nachr_conductance_final_ns": renshaw_nachr_conductance,
        "renshaw_glutamate_conductance_final_ns": renshaw_glutamate_conductance,
        "renshaw_h_gate_final": renshaw_h_gate,
        "renshaw_calcium_final": renshaw_calcium,
        "v3_t_inactivation_final": v3_t_inactivation,
        "mn_dendrite_voltage_final_mv": mn_dendrite_v[is_mn],
        "mn_nap_pic_inactivation_final": mn_nap_pic_inactivation[is_mn],
        "mn_ltype_ca_pic_activation_final": mn_ltype_ca_pic_activation[is_mn],
        "mechanism_component_names": mechanism_component_names,
        "mechanism_component_mean_abs_pa": (
            mechanism_abs_sum / max(mechanism_observation_count, 1)
        ),
        "renshaw_mixed_input_current_names": np.asarray([
            "I_RENSHAW_NACHR", "I_RENSHAW_GLUTAMATE",
        ]),
        "renshaw_mixed_input_current_mean_abs_pa": (
            renshaw_mixed_input_abs_sum / max(mechanism_observation_count, 1)
        ),
        "edge_source_side": edge_side_index,
        "local_mt_state_count": np.asarray([
            int(np.sum(resource_edge_mask))
        ]),
        "kca_replay_uses_physical_time": np.asarray([kca_time_api]),
        "kca_replay_event_count_by_neuron": np.bincount(
            kca_event_neurons, minlength=n_neuron
        ).astype(np.int64),
        "mt_replay_uses_physical_time": np.asarray([mt_time_api]),
        "mt_replay_event_count_by_edge": np.bincount(
            mt_event_edges, minlength=len(edge_rrp_available)
        ).astype(np.int64),
        "seed": np.asarray([seed]),
        "structural_seed": np.asarray([structural_seed]),
        "protocol": np.asarray([protocol]),
        "mt_mode": np.asarray([mt_mode]),
        "static_scale": np.asarray([static_scale]),
        "fast_mode": np.asarray([fast_mode]),
        "model_version": np.asarray([MODEL_VERSION]),
        "ablated_pathways": np.asarray(sorted(set(ablated_pathways)), dtype="U40"),
        "disabled_intrinsic_mechanisms": np.asarray(
            sorted(disabled_intrinsic_mechanism_set), dtype="U40"
        ),
        "ablated_populations": np.asarray(sorted(set(ablated_populations)), dtype="U40"),
        "speed_level": np.asarray([speed_level]),
        "load_context": np.asarray([load_context]),
        "load_side": np.asarray([load_side]),
        "pulse_direction": np.asarray([pulse_direction]),
        "pulse_cycle_fraction_override": np.asarray([
            np.nan if pulse_cycle_fraction_override is None
            else pulse_cycle_fraction_override
        ]),
        "pulse_target_side": np.asarray([pulse_target_side]),
        "pulse_target_phase": np.asarray([pulse_target_phase]),
        "pulse_start_s": np.asarray([
            pulse_start_step * dt / 1000.0 if pulse_start_step >= 0 else np.nan
        ]),
        "pulse_end_s": np.asarray([
            pulse_end_step * dt / 1000.0 if pulse_end_step >= 0 else np.nan
        ]),
        "sham_excitatory_start_s": np.asarray([
            sham_excitatory_start_step * dt / 1000.0
            if sham_excitatory_start_step >= 0 else np.nan
        ]),
        "sham_excitatory_end_s": np.asarray([
            sham_excitatory_end_step * dt / 1000.0
            if sham_excitatory_end_step >= 0 else np.nan
        ]),
        "sham_inhibitory_start_s": np.asarray([
            sham_inhibitory_start_step * dt / 1000.0
            if sham_inhibitory_start_step >= 0 else np.nan
        ]),
        "sham_inhibitory_end_s": np.asarray([
            sham_inhibitory_end_step * dt / 1000.0
            if sham_inhibitory_end_step >= 0 else np.nan
        ]),
        "impaired_mt_routes": np.asarray(sorted(set(impaired_mt_routes)), dtype="U16"),
        "challenged_routes": np.asarray(sorted(set(challenged_routes)), dtype="U16"),
        "fast_activation_scale": np.asarray([fast_activation_scale]),
    }


def build_yoked_kca_event_times(
    reference_trace: Mapping[str, np.ndarray],
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic KCa replay as physical event times and neuron ids."""
    sizes, _, _, _ = population_metadata(cfg)
    offsets = np.cumsum(np.r_[0, sizes[:-1]])
    pop_ids = np.asarray(reference_trace["spike_population"], dtype=int)
    local_ids = np.asarray(reference_trace["spike_neuron"], dtype=int)
    times = np.asarray(reference_trace["spike_time_s"], dtype=float)
    if not (pop_ids.shape == local_ids.shape == times.shape):
        raise ValueError("reference spike arrays must have identical shape")
    global_ids = offsets[pop_ids] + local_ids
    positive = np.asarray(reference_trace["rg_kca_positive_mask"], dtype=bool)
    if positive.shape != (int(np.sum(sizes)),):
        raise ValueError("reference KCa-positive mask has wrong shape")
    rg_mask = np.asarray([POPULATIONS[p].startswith("RG_") for p in pop_ids])
    eligible_event = rg_mask & positive[global_ids]
    global_ids, times = global_ids[eligible_event], times[eligible_event]
    rng = np.random.default_rng(seed + 440044)
    mapping: Dict[int, int] = {}
    shift: Dict[int, float] = {}
    for side in SIDES:
        side_pops = [pop("RG", side, phase) for phase in PHASES]
        cells = np.concatenate([
            offsets[p] + np.arange(sizes[p], dtype=int) for p in side_pops
        ])
        cells = cells[positive[cells]]
        permuted = rng.permutation(cells)
        for source, target in zip(cells, permuted):
            mapping[int(source)] = int(target)
            shift[int(source)] = float(
                rng.uniform(0.25, max(0.251, cfg.duration_s - 0.25))
            )
    yoked_neurons = np.asarray(
        [mapping[int(cell)] for cell in global_ids], dtype=np.int64
    )
    yoked_times = np.asarray([
        (time + shift[int(cell)]) % cfg.duration_s
        for time, cell in zip(times, global_ids)
    ], dtype=float)
    order = np.argsort(yoked_times, kind="stable")
    return yoked_times[order], yoked_neurons[order]


def build_yoked_kca_events(
    reference_trace: Mapping[str, np.ndarray],
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy/debug integer-step KCa replay builder.

    Each RG cell's event train receives a deterministic circular time shift and
    is reassigned to another RG cell on the same side. Event count is exact;
    target phase/cell identity and alignment to the receiving cell are broken.
    """
    sizes, _, _, _ = population_metadata(cfg)
    offsets = np.cumsum(np.r_[0, sizes[:-1]])
    pop_ids = np.asarray(reference_trace["spike_population"], dtype=int)
    local_ids = np.asarray(reference_trace["spike_neuron"], dtype=int)
    times = np.asarray(reference_trace["spike_time_s"], dtype=float)
    global_ids = offsets[pop_ids] + local_ids
    positive = np.asarray(reference_trace["rg_kca_positive_mask"], dtype=bool)
    if positive.shape != (int(np.sum(sizes)),):
        raise ValueError("reference KCa-positive mask has wrong shape")
    rg_mask = np.asarray([POPULATIONS[p].startswith("RG_") for p in pop_ids])
    eligible_event = rg_mask & positive[global_ids]
    global_ids, times = global_ids[eligible_event], times[eligible_event]
    rng = np.random.default_rng(seed + 440044)
    mapping: Dict[int, int] = {}
    shift: Dict[int, float] = {}
    for side in SIDES:
        side_pops = [pop("RG", side, phase) for phase in PHASES]
        cells = np.concatenate([
            offsets[p] + np.arange(sizes[p], dtype=int) for p in side_pops
        ])
        cells = cells[positive[cells]]
        permuted = rng.permutation(cells)
        for source, target in zip(cells, permuted):
            mapping[int(source)] = int(target)
            shift[int(source)] = float(rng.uniform(0.25, max(0.251, cfg.duration_s - 0.25)))
    yoked_neurons = np.asarray([mapping[int(cell)] for cell in global_ids], dtype=np.int64)
    yoked_times = np.asarray([
        (time + shift[int(cell)]) % cfg.duration_s for time, cell in zip(times, global_ids)
    ])
    yoked_steps = np.floor(yoked_times * 1000.0 / cfg.dt_ms).astype(np.int64)
    order = np.argsort(yoked_steps, kind="stable")
    return yoked_steps[order], yoked_neurons[order]


def build_yoked_mt_event_times(
    reference_trace: Mapping[str, np.ndarray],
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic terminal replay at continuous physical times.

    Every eligible intact terminal retains its reference event count and
    inter-event structure under a terminal-specific circular time shift. The
    shift is sampled in seconds, so replay construction does not quantize an
    interpolated spike time to a simulator step.
    """
    if "mt_mode" in reference_trace and str(reference_trace["mt_mode"][0]) != "dynamic":
        raise ValueError("MT yoked events require a dynamic-MT reference trace")
    sizes = np.asarray(reference_trace["population_sizes"], dtype=int)
    expected_sizes = population_metadata(cfg)[0]
    if not np.array_equal(sizes, expected_sizes):
        raise ValueError("reference population sizes do not match configuration")
    offsets = np.cumsum(np.r_[0, sizes[:-1]])
    pop_ids = np.asarray(reference_trace["spike_population"], dtype=int)
    local_ids = np.asarray(reference_trace["spike_neuron"], dtype=int)
    times = np.asarray(reference_trace["spike_time_s"], dtype=float)
    if not (pop_ids.shape == local_ids.shape == times.shape):
        raise ValueError("reference spike arrays must have identical shape")
    global_ids = offsets[pop_ids] + local_ids
    times_by_source: Dict[int, np.ndarray] = {}
    if len(global_ids):
        order = np.argsort(global_ids, kind="stable")
        sorted_ids = global_ids[order]
        split_at = np.flatnonzero(np.diff(sorted_ids)) + 1
        for group in np.split(order, split_at):
            times_by_source[int(global_ids[group[0]])] = times[group]

    sources = np.asarray(reference_trace["connectome_source"], dtype=int)
    routes = np.asarray(reference_trace["connectome_mt_route"]).astype(str)
    ablated = np.asarray(
        reference_trace.get(
            "connectome_ablated_edge_mask",
            np.zeros(len(sources), dtype=bool),
        ),
        dtype=bool,
    )
    if not (sources.shape == routes.shape == ablated.shape):
        raise ValueError("reference edge arrays must have identical shape")
    eligible_edges = np.flatnonzero((routes != "none") & ~ablated)
    rng = np.random.default_rng(seed + 550055)
    replay_time_chunks: List[np.ndarray] = []
    replay_edge_chunks: List[np.ndarray] = []
    for edge in eligible_edges:
        source_times = times_by_source.get(int(sources[edge]))
        if source_times is None or not len(source_times):
            continue
        shift_s = float(rng.uniform(0.0, cfg.duration_s))
        replay_time_chunks.append((source_times + shift_s) % cfg.duration_s)
        replay_edge_chunks.append(
            np.full(len(source_times), int(edge), dtype=np.int64)
        )
    if not replay_time_chunks:
        return np.asarray([], dtype=float), np.asarray([], dtype=np.int64)
    replay_times = np.concatenate(replay_time_chunks).astype(float, copy=False)
    replay_edges = np.concatenate(replay_edge_chunks).astype(np.int64, copy=False)
    order = np.argsort(replay_times, kind="stable")
    return replay_times[order], replay_edges[order]


def build_yoked_mt_events(
    reference_trace: Mapping[str, np.ndarray],
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create legacy/debug integer-step MT replay from a dynamic reference.

    Every eligible presynaptic terminal receives exactly as many activity
    events as it received in the dynamic reference.  A deterministic,
    terminal-specific circular time shift preserves event count and each
    terminal's inter-event structure while breaking alignment to ongoing
    spikes in the counterfactual simulation.  No outcome variable is read.
    """
    if "mt_mode" in reference_trace and str(reference_trace["mt_mode"][0]) != "dynamic":
        raise ValueError("MT yoked events require a dynamic-MT reference trace")
    sizes = np.asarray(reference_trace["population_sizes"], dtype=int)
    expected_sizes = population_metadata(cfg)[0]
    if not np.array_equal(sizes, expected_sizes):
        raise ValueError("reference population sizes do not match configuration")
    offsets = np.cumsum(np.r_[0, sizes[:-1]])
    pop_ids = np.asarray(reference_trace["spike_population"], dtype=int)
    local_ids = np.asarray(reference_trace["spike_neuron"], dtype=int)
    times = np.asarray(reference_trace["spike_time_s"], dtype=float)
    if not (pop_ids.shape == local_ids.shape == times.shape):
        raise ValueError("reference spike arrays must have identical shape")
    global_ids = offsets[pop_ids] + local_ids
    n_step = int(round(cfg.duration_s * 1000.0 / cfg.dt_ms))
    spike_steps = np.rint(times * 1000.0 / cfg.dt_ms).astype(np.int64)
    spike_steps = np.clip(spike_steps, 0, max(n_step - 1, 0))
    steps_by_source: Dict[int, np.ndarray] = {}
    if len(global_ids):
        order = np.argsort(global_ids, kind="stable")
        sorted_ids = global_ids[order]
        split_at = np.flatnonzero(np.diff(sorted_ids)) + 1
        for group in np.split(order, split_at):
            steps_by_source[int(global_ids[group[0]])] = spike_steps[group]

    sources = np.asarray(reference_trace["connectome_source"], dtype=int)
    routes = np.asarray(reference_trace["connectome_mt_route"]).astype(str)
    ablated = np.asarray(
        reference_trace.get(
            "connectome_ablated_edge_mask",
            np.zeros(len(sources), dtype=bool),
        ),
        dtype=bool,
    )
    if not (sources.shape == routes.shape == ablated.shape):
        raise ValueError("reference edge arrays must have identical shape")
    eligible_edges = np.flatnonzero((routes != "none") & ~ablated)
    rng = np.random.default_rng(seed + 550055)
    replay_step_chunks: List[np.ndarray] = []
    replay_edge_chunks: List[np.ndarray] = []
    for edge in eligible_edges:
        source_steps = steps_by_source.get(int(sources[edge]))
        if source_steps is None or not len(source_steps):
            continue
        shift = int(rng.integers(1, n_step)) if n_step > 1 else 0
        replay_step_chunks.append((source_steps + shift) % max(n_step, 1))
        replay_edge_chunks.append(
            np.full(len(source_steps), int(edge), dtype=np.int64)
        )
    if not replay_step_chunks:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    replay_steps = np.concatenate(replay_step_chunks).astype(np.int64, copy=False)
    replay_edges = np.concatenate(replay_edge_chunks).astype(np.int64, copy=False)
    order = np.argsort(replay_steps, kind="stable")
    return replay_steps[order], replay_edges[order]


def simulation_manifest(trace: Mapping[str, np.ndarray], cfg: Config) -> Dict[str, object]:
    """Machine-verifiable identity for every reported simulation."""
    parameter_json = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
    equation_json = json.dumps(
        {name: asdict(record) for name, record in CELL_CLASS_EQUATIONS.items()},
        sort_keys=True, separators=(",", ":"),
    )
    interface_json = json.dumps(
        {name: asdict(record) for name, record in BIOLOGICAL_INTERFACE_EQUATIONS.items()},
        sort_keys=True, separators=(",", ":"),
    )
    execution_contract_json = json.dumps(
        {name: asdict(record) for name, record in CLASS_EXECUTION_CONTRACTS.items()},
        sort_keys=True, separators=(",", ":"),
    )
    literature_contract_json = json.dumps(
        literature_evidence_contract_payload(),
        sort_keys=True, separators=(",", ":"),
    )
    code_bytes = Path(__file__).read_bytes()
    return {
        "model_version": str(trace["model_version"][0]),
        "model_sha256": hashlib.sha256(code_bytes).hexdigest(),
        "parameter_sha256": hashlib.sha256(parameter_json.encode()).hexdigest(),
        "cell_class_equation_sha256": hashlib.sha256(equation_json.encode()).hexdigest(),
        "biological_interface_equation_sha256": hashlib.sha256(interface_json.encode()).hexdigest(),
        "class_execution_contract_sha256": hashlib.sha256(
            execution_contract_json.encode()
        ).hexdigest(),
        "literature_evidence_contract_sha256": hashlib.sha256(
            literature_contract_json.encode()
        ).hexdigest(),
        "parameters": asdict(cfg),
        "seed": int(trace["seed"][0]),
        "structural_seed": int(trace["structural_seed"][0]),
        "protocol": str(trace["protocol"][0]),
        "fast_mode": str(trace["fast_mode"][0]),
        "mt_mode": str(trace["mt_mode"][0]),
        "fast_activation_scale": float(trace["fast_activation_scale"][0]),
        "ablated_pathways": trace["ablated_pathways"].tolist(),
        "disabled_intrinsic_mechanisms": trace[
            "disabled_intrinsic_mechanisms"
        ].tolist(),
        "ablated_populations": trace["ablated_populations"].tolist(),
        "speed_level": str(trace["speed_level"][0]),
        "load_context": str(trace["load_context"][0]),
        "load_side": str(trace["load_side"][0]),
        "pulse_direction": str(trace["pulse_direction"][0]),
        "pulse_cycle_fraction_override": float(
            trace["pulse_cycle_fraction_override"][0]
        ),
        "pulse_target": f"{trace['pulse_target_side'][0]}-{trace['pulse_target_phase'][0]}",
        "pulse_start_s": float(trace["pulse_start_s"][0]),
        "pulse_end_s": float(trace["pulse_end_s"][0]),
        "sham_excitatory_start_s": float(
            trace["sham_excitatory_start_s"][0]
        ),
        "sham_inhibitory_start_s": float(
            trace["sham_inhibitory_start_s"][0]
        ),
        "impaired_mt_routes": trace["impaired_mt_routes"].tolist(),
        "challenged_routes": trace["challenged_routes"].tolist(),
        "local_presynaptic_terminal_mt": True,
        "mt_routes_defined_by_presynaptic_identity": True,
        "mt_only_modulates_slow_vesicle_replenishment": True,
        "mt_damage_or_repair_coupling": False,
        "primary_endpoints_are_hidden_state_free": True,
        "local_mt_state_count": int(trace["local_mt_state_count"][0]),
        "speed_dependent_synaptic_rescaling": False,
        "class_specific_speed_drive_offsets": False,
        "load_enters_mechanics_only": True,
        "load_is_external_resistance_not_active_force_gain": True,
        "generic_relay_equation_fallback": False,
        "all_declared_classes_have_explicit_full_dynamical_identities": True,
        "all_declared_classes_have_unique_intrinsic_channel_models": False,
        "unsupported_unique_intrinsic_channel_claims": False,
        "unsupported_model_only_subtypes_removed": True,
        "explicit_nmj_dynamics": True,
        "direct_mn_rate_to_muscle_activation": False,
        "within_step_spike_crossing_interpolated": True,
        "membrane_integrator": "local_exponential_rosenbrock_with_event_time_inversion",
        "refractory_adaptation_substep_exact": True,
        "remaining_hybrid_reduction": (
            "slow gates and continuous mediator states are frozen or advanced "
            "once over each outer step; spike-triggered jumps are event-aged"
        ),
        "all_named_biological_interfaces_have_equations": True,
        "population_count": int(len(POPULATIONS)),
        "spiking_neuron_count": int(np.sum(trace["population_sizes"])),
        "edge_count": int(len(trace["connectome_source"])),
        "numpy_version": np.__version__,
    }


def json_safe(value: object) -> object:
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def write_simulation_manifest(path: Path, trace: Mapping[str, np.ndarray], cfg: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(simulation_manifest(trace, cfg)), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def detect_population_bursts(
    trace: Mapping[str, np.ndarray],
    cfg: Config,
    cell_class: str = "RG",
) -> Dict[str, np.ndarray]:
    """Detect population burst onsets with class-specific hysteresis."""
    if cell_class not in {"RG", "PF", "MN"}:
        raise ValueError("Burst detection is defined only for RG, PF and MN populations")
    if cell_class == "MN":
        on_threshold = cfg.mn_burst_on_threshold_hz
        off_threshold = cfg.mn_burst_off_threshold_hz
    elif cell_class == "PF":
        on_threshold = cfg.pf_burst_on_threshold_hz
        off_threshold = cfg.pf_burst_off_threshold_hz
    else:
        on_threshold = cfg.burst_on_threshold_hz
        off_threshold = cfg.burst_off_threshold_hz
    result: Dict[str, np.ndarray] = {}
    time = trace["time_s"]
    rate = trace["rate_hz"]
    for side, phase in SIDE_PHASES:
        name = population_name(cell_class, side, phase)
        values = rate[:, pop(cell_class, side, phase)]
        onsets: List[float] = []
        armed = True
        last = -np.inf
        for index in range(1, len(values)):
            if armed and values[index - 1] < on_threshold <= values[index]:
                event_time = float(time[index])
                if event_time - last >= cfg.minimum_interburst_s:
                    onsets.append(event_time)
                    last = event_time
                    armed = False
            elif not armed and values[index] <= off_threshold:
                armed = True
        result[name] = np.asarray(onsets)
    return result


def cycle_phase_errors_deg(a: np.ndarray, b: np.ndarray, start_s: float) -> np.ndarray:
    errors: List[float] = []
    for t0, t1 in zip(a[:-1], a[1:]):
        if t0 < start_s:
            continue
        inside = b[(b > t0) & (b < t1)]
        if len(inside) == 1:
            fraction = (inside[0] - t0) / (t1 - t0)
            errors.append((fraction - 0.5) * 360.0)
        else:
            # Zero or multiple counterpart bursts both violate one-to-one
            # alternation and are retained as biological phase slips.
            errors.append(180.0)
    return np.asarray(errors)


def cycle_interval_cv(events: np.ndarray, start_s: float) -> float:
    selected = events[events >= start_s]
    if len(selected) < 4:
        return float("nan")
    intervals = np.diff(selected)
    return float(np.std(intervals, ddof=1) / np.mean(intervals))


def mean_frequency_in_window(events: np.ndarray, start_s: float, end_s: float) -> float:
    selected = events[(events >= start_s) & (events <= end_s)]
    return float(1.0 / np.mean(np.diff(selected))) if len(selected) >= 3 else float("nan")


def rg_pf_latencies(
    rg_events: np.ndarray,
    pf_events: np.ndarray,
    start_s: float,
    end_s: float,
    window_s: float,
) -> Tuple[np.ndarray, int]:
    """Match each RG onset to the first homologous PF onset after it."""
    selected_rg = rg_events[(rg_events >= start_s) & (rg_events <= end_s)]
    latencies: List[float] = []
    missed = 0
    for onset in selected_rg:
        index = int(np.searchsorted(pf_events, onset, side="left"))
        if index < len(pf_events) and pf_events[index] <= onset + window_s:
            latencies.append(float(pf_events[index] - onset))
        else:
            missed += 1
    return np.asarray(latencies), missed


def match_output_bursts(
    rg_events: np.ndarray,
    output_events: np.ndarray,
    start_s: float,
    end_s: float,
    pre_window_s: float,
    post_window_s: float,
) -> Tuple[np.ndarray, int, int]:
    """One-to-one, maximum-cardinality monotone transfer matching.

    Anchors are eligible only when their complete pre/post observation window
    lies inside the half-open analysis interval. Selecting the earliest
    admissible unused output preserves maximum cardinality for these ordered,
    equal-width windows; nearest-neighbor greed can consume the only event
    available to a later anchor.
    """
    rg_events = np.asarray(rg_events, dtype=float)
    output_events = np.asarray(output_events, dtype=float)
    scalar_inputs = (start_s, end_s, pre_window_s, post_window_s)
    if not all(math.isfinite(float(value)) for value in scalar_inputs):
        raise ValueError("transfer-match inputs must be finite")
    if pre_window_s < 0.0 or post_window_s < 0.0:
        raise ValueError("transfer-match windows must be nonnegative")
    if end_s <= start_s:
        raise ValueError("transfer-match interval must have positive duration")
    for values, label in ((rg_events, "RG"), (output_events, "output")):
        if values.ndim != 1 or np.any(~np.isfinite(values)):
            raise ValueError(f"{label} burst events must be a finite 1-D array")
        if np.any(np.diff(values) < 0.0):
            raise ValueError(f"{label} burst events must be sorted")
    anchors = rg_events[
        (rg_events >= start_s + pre_window_s)
        & (rg_events < end_s - post_window_s)
    ]
    latencies: List[float] = []
    missed = 0
    output_cursor = 0
    for onset in anchors:
        lower = onset - pre_window_s
        upper = onset + post_window_s
        while output_cursor < len(output_events) and output_events[output_cursor] < lower:
            output_cursor += 1
        if output_cursor < len(output_events) and output_events[output_cursor] <= upper:
            latencies.append(float(output_events[output_cursor] - onset))
            output_cursor += 1
        else:
            missed += 1
    return np.asarray(latencies), missed, int(len(anchors))


def right_endpoint_sample_mask(
    time_s: np.ndarray, start_s: float, end_s: float
) -> np.ndarray:
    """Select continuous samples labeled by integration right endpoints."""
    values = np.asarray(time_s, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("right-endpoint sample times must be a finite 1-D array")
    if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
        raise ValueError("right-endpoint interval must be finite and nonempty")
    return (values > start_s) & (values <= end_s)


def phase_relation_events(
    anchor: np.ndarray,
    counterpart: np.ndarray,
    start_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    errors: List[float] = []
    for t0, t1 in zip(anchor[:-1], anchor[1:]):
        if t0 < start_s:
            continue
        inside = counterpart[(counterpart > t0) & (counterpart < t1)]
        times.append(float(t0))
        if len(inside) == 1:
            fraction = float((inside[0] - t0) / (t1 - t0))
            errors.append((fraction - 0.5) * 360.0)
        else:
            errors.append(180.0)
    return np.asarray(times), np.asarray(errors)


def _safe_concat(values: Sequence[np.ndarray]) -> np.ndarray:
    nonempty = [np.asarray(value) for value in values if len(value)]
    return np.concatenate(nonempty) if nonempty else np.asarray([])


def recovery_outcome_from_phase_and_period(
    bursts: Mapping[str, np.ndarray],
    cfg: Config,
    pulse_start_s: float,
    pulse_end_s: float,
) -> Dict[str, object]:
    """Return an event/censor record for recovery of all four RG relations.

    Recovery requires both left-right pairs, both flexor-extensor pairs and
    all four RG population periods to satisfy prespecified tolerances for the
    prespecified number of consecutive cycles. Phase uses the single frozen
    ``phase_slip_threshold_deg`` in every arm; pre-pulse phase errors are read
    only to establish endpoint eligibility. Failure to recover by the end of
    the simulation is right censored rather than encoded as NaN.
    """
    empty = {
        "recovery_time_s": float("nan"),
        "recovery_event_observed": 0,
        "recovery_time_or_censor_s": float("nan"),
        "recovery_censor_time_s": float("nan"),
        "recovery_endpoint_eligible": 0,
        "recovery_ineligibility_reason": "pulse_not_delivered",
    }
    if not np.isfinite(pulse_start_s) or not np.isfinite(pulse_end_s):
        return empty
    censor_time = max(0.0, cfg.duration_s - pulse_end_s)
    if pulse_end_s >= cfg.duration_s:
        return {
            **empty,
            "recovery_censor_time_s": censor_time,
            "recovery_ineligibility_reason": "pulse_ended_at_or_after_simulation_end",
        }

    lf = bursts[population_name("RG", "L", "F")]
    le = bursts[population_name("RG", "L", "E")]
    rf = bursts[population_name("RG", "R", "F")]
    re = bursts[population_name("RG", "R", "E")]
    populations = (lf, le, rf, re)
    relation_pairs = (
        (lf, rf),  # flexor left-right
        (le, re),  # extensor left-right
        (lf, le),  # left flexor-extensor
        (rf, re),  # right flexor-extensor
    )
    relation_data = [
        phase_relation_events(anchor, counterpart, cfg.burn_in_s)
        for anchor, counterpart in relation_pairs
    ]
    pre_errors = [
        np.abs(errors[(times < pulse_start_s)])
        for times, errors in relation_data
    ]
    pre_events = [
        events[(events >= cfg.burn_in_s) & (events < pulse_start_s)]
        for events in populations
    ]
    if any(len(values) < 2 for values in pre_errors) or any(
        len(events) < 3 for events in pre_events
    ):
        return {
            **empty,
            "recovery_censor_time_s": censor_time,
            "recovery_ineligibility_reason": "insufficient_prepulse_cycles",
        }

    periods = [float(np.median(np.diff(events))) for events in pre_events]
    n = cfg.recovery_consecutive_cycles
    post_lf = lf[lf >= pulse_end_s]
    for candidate in post_lf:
        horizon = candidate + (n + 1.5) * max(periods)
        relation_ok = True
        for times, errors in relation_data:
            selected = np.abs(errors[(times >= candidate) & (times <= horizon)])[:n]
            if len(selected) < n or np.any(
                selected > cfg.phase_slip_threshold_deg
            ):
                relation_ok = False
                break
        if not relation_ok:
            continue
        period_ok = True
        for events, period in zip(populations, periods):
            selected = events[events >= candidate][:n + 1]
            intervals = np.diff(selected)
            if len(intervals) < n or np.any(
                np.abs(intervals - period)
                > cfg.recovery_frequency_tolerance_fraction * period
            ):
                period_ok = False
                break
        if relation_ok and period_ok:
            observed = float(candidate - pulse_end_s)
            return {
                "recovery_time_s": observed,
                "recovery_event_observed": 1,
                "recovery_time_or_censor_s": observed,
                "recovery_censor_time_s": censor_time,
                "recovery_endpoint_eligible": 1,
                "recovery_ineligibility_reason": "none",
            }
    return {
        "recovery_time_s": float("nan"),
        "recovery_event_observed": 0,
        "recovery_time_or_censor_s": censor_time,
        "recovery_censor_time_s": censor_time,
        "recovery_endpoint_eligible": 1,
        "recovery_ineligibility_reason": "right_censored_no_recovery",
    }


def recovery_time_from_phase_and_period(
    bursts: Mapping[str, np.ndarray],
    cfg: Config,
    pulse_start_s: float,
    pulse_end_s: float,
) -> float:
    """Backward-compatible observed recovery time (NaN when censored)."""
    return float(recovery_outcome_from_phase_and_period(
        bursts, cfg, pulse_start_s, pulse_end_s
    )["recovery_time_s"])


def technical_trace_quality(trace: Mapping[str, np.ndarray]) -> Dict[str, object]:
    """Classify numerical/file-level validity without censoring biology."""
    allowed_nonfinite = {
        "pulse_cycle_fraction_override", "pulse_start_s", "pulse_end_s",
        "sham_excitatory_start_s", "sham_excitatory_end_s",
        "sham_inhibitory_start_s", "sham_inhibitory_end_s",
    }
    invalid: List[str] = []
    for key, value in trace.items():
        array = np.asarray(value)
        if array.dtype.kind not in {"f", "c"} or key in allowed_nonfinite:
            continue
        if not np.all(np.isfinite(array)):
            invalid.append(key)
    return {
        "technical_valid": int(not invalid),
        "technical_exclusion_reason": (
            "none" if not invalid else "nonfinite_state:" + "+".join(sorted(invalid))
        ),
    }


def summarize(trace: Mapping[str, np.ndarray], cfg: Config) -> Dict[str, object]:
    rg_bursts = detect_population_bursts(trace, cfg, "RG")
    pf_bursts = detect_population_bursts(trace, cfg, "PF")
    mn_bursts = detect_population_bursts(trace, cfg, "MN")
    quality = technical_trace_quality(trace)
    clean = {name: values[values >= cfg.burn_in_s] for name, values in rg_bursts.items()}
    frequencies = [
        1.0 / np.mean(np.diff(values)) for values in clean.values()
        if len(values) >= 2 and np.mean(np.diff(values)) > 0
    ]
    left_f = clean[population_name("RG", "L", "F")]
    right_f = clean[population_name("RG", "R", "F")]
    left_e = clean[population_name("RG", "L", "E")]
    right_e = clean[population_name("RG", "R", "E")]
    lr_sets = (
        cycle_phase_errors_deg(left_f, right_f, cfg.burn_in_s),
        cycle_phase_errors_deg(left_e, right_e, cfg.burn_in_s),
    )
    fe_sets = (
        cycle_phase_errors_deg(left_f, left_e, cfg.burn_in_s),
        cycle_phase_errors_deg(right_f, right_e, cfg.burn_in_s),
    )
    lr_errors = _safe_concat(lr_sets)
    fe_errors = _safe_concat(fe_sets)
    pulse_start = float(np.asarray(trace.get("pulse_start_s", [np.nan]))[0])
    pulse_end = float(np.asarray(trace.get("pulse_end_s", [np.nan]))[0])
    sham_excitatory_start = float(np.asarray(
        trace.get("sham_excitatory_start_s", [np.nan])
    )[0])
    sham_inhibitory_start = float(np.asarray(
        trace.get("sham_inhibitory_start_s", [np.nan])
    )[0])

    def phase_errors_after(start_s: float) -> Tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(start_s):
            return np.asarray([]), np.asarray([])
        return (
            _safe_concat((
                cycle_phase_errors_deg(left_f, right_f, start_s),
                cycle_phase_errors_deg(left_e, right_e, start_s),
            )),
            _safe_concat((
                cycle_phase_errors_deg(left_f, left_e, start_s),
                cycle_phase_errors_deg(right_f, right_e, start_s),
            )),
        )

    post_lr, post_fe = phase_errors_after(pulse_start)
    sham_excitatory_lr, sham_excitatory_fe = phase_errors_after(
        sham_excitatory_start
    )
    sham_inhibitory_lr, sham_inhibitory_fe = phase_errors_after(
        sham_inhibitory_start
    )
    cycle_cvs = np.asarray([
        cycle_interval_cv(values, cfg.burn_in_s) for values in clean.values()
    ])
    cycle_cvs = cycle_cvs[np.isfinite(cycle_cvs)]
    pre_frequencies = [
        mean_frequency_in_window(values, cfg.burn_in_s, cfg.perturbation_start_s)
        for values in clean.values()
    ]
    post_window_start = min(cfg.duration_s - 0.5, cfg.perturbation_end_s + 0.25)
    post_frequencies = [
        mean_frequency_in_window(values, post_window_start, cfg.duration_s)
        for values in clean.values()
    ]
    pre_frequencies = [value for value in pre_frequencies if np.isfinite(value)]
    post_frequencies = [value for value in post_frequencies if np.isfinite(value)]
    pre_frequency = float(np.mean(pre_frequencies)) if pre_frequencies else float("nan")
    post_frequency = float(np.mean(post_frequencies)) if post_frequencies else float("nan")
    relay_rates: Dict[str, float] = {}
    time_mask = trace["time_s"] >= cfg.burn_in_s
    for cell_class in CLASSES:
        indices = [pop(cell_class, side, phase) for side, phase in SIDE_PHASES]
        relay_rates[f"{cell_class}_mean_rate_hz"] = float(np.mean(trace["rate_hz"][time_mask][:, indices]))

    pf_latencies: List[float] = []
    mn_latencies: List[float] = []
    total_rg_pf = total_pf_missed = 0
    total_rg_mn = total_mn_missed = 0
    for side, phase in SIDE_PHASES:
        rg_name = population_name("RG", side, phase)
        pf_name = population_name("PF", side, phase)
        mn_name = population_name("MN", side, phase)
        latency, missed, total = match_output_bursts(
            rg_bursts[rg_name], pf_bursts[pf_name], cfg.burn_in_s,
            cfg.duration_s, 0.0, cfg.rg_pf_match_window_s,
        )
        pf_latencies.extend(latency.tolist())
        total_pf_missed += missed
        total_rg_pf += total
        latency, missed, total = match_output_bursts(
            rg_bursts[rg_name], mn_bursts[mn_name], cfg.burn_in_s,
            cfg.duration_s, cfg.rg_mn_match_pre_window_s,
            cfg.rg_mn_match_post_window_s,
        )
        mn_latencies.extend(latency.tolist())
        total_mn_missed += missed
        total_rg_mn += total

    pf_latency_array = np.asarray(pf_latencies)
    mn_latency_array = np.asarray(mn_latencies)
    mn_indices_left = [pop("MN", "L", phase) for phase in PHASES]
    mn_indices_right = [pop("MN", "R", phase) for phase in PHASES]
    left_amplitude = float(np.mean(np.sum(trace["rate_hz"][time_mask][:, mn_indices_left], axis=1)))
    right_amplitude = float(np.mean(np.sum(trace["rate_hz"][time_mask][:, mn_indices_right], axis=1)))
    amplitude_denominator = left_amplitude + right_amplitude
    imbalance = abs(left_amplitude - right_amplitude) / amplitude_denominator if amplitude_denominator > 0 else float("nan")
    recovery_outcome = recovery_outcome_from_phase_and_period(
        rg_bursts, cfg, pulse_start, pulse_end
    )
    burst_counts = {
        f"n_bursts_{side}_{phase}": int(
            len(clean[population_name("RG", side, phase)])
        )
        for side, phase in SIDE_PHASES
    }
    rhythmic_failure = int(min(burst_counts.values()) < 2)
    pulse_required = (
        str(np.asarray(trace["protocol"])[0]) == "pulse"
        and str(np.asarray(trace["pulse_direction"])[0]) != "none"
    )
    pulse_delivered = int(
        np.isfinite(pulse_start) and np.isfinite(pulse_end)
        and pulse_end <= cfg.duration_s
    )

    def baseline_cycle_count(anchor_s: float) -> int:
        if not np.isfinite(anchor_s):
            return 0
        return min(
            max(0, int(np.sum(
                (events >= cfg.burn_in_s) & (events < anchor_s)
            )) - 1)
            for events in rg_bursts.values()
        )

    baseline_cycles_before_pulse = baseline_cycle_count(pulse_start)
    baseline_cycles_before_excitatory_sham = baseline_cycle_count(
        sham_excitatory_start
    )
    baseline_cycles_before_inhibitory_sham = baseline_cycle_count(
        sham_inhibitory_start
    )
    pulse_response_eligible = int(
        bool(quality["technical_valid"])
        and pulse_required and pulse_delivered
        and baseline_cycles_before_pulse >= 2
    )
    pulse_noneligibility_reason = "none"
    if pulse_required and not pulse_delivered:
        pulse_noneligibility_reason = "biological_no_phase_eligible_cycle"
    elif pulse_required and baseline_cycles_before_pulse < 2:
        pulse_noneligibility_reason = "insufficient_prepulse_cycles"
    elif not pulse_required:
        pulse_noneligibility_reason = "no_pulse_condition"
    elif not quality["technical_valid"]:
        pulse_noneligibility_reason = str(quality["technical_exclusion_reason"])
    if not pulse_required or not quality["technical_valid"]:
        recovery_composite_eligible = 0
        recovery_composite_event = 0
        recovery_composite_time_s = float("nan")
    elif pulse_delivered and recovery_outcome["recovery_endpoint_eligible"]:
        recovery_composite_eligible = 1
        recovery_composite_event = int(
            recovery_outcome["recovery_event_observed"]
        )
        recovery_composite_time_s = float(
            recovery_outcome["recovery_time_or_censor_s"]
        )
    else:
        # Failure to establish a phase-eligible cycle is a biological failure,
        # not a technical exclusion. Retain it as non-recovery through the
        # maximum post-arm observation horizon.
        recovery_composite_eligible = 1
        recovery_composite_event = 0
        recovery_composite_time_s = max(
            0.0, cfg.duration_s - cfg.pulse_arm_after_s
        )
    route_support = np.asarray(trace.get("mt_route_support", []), dtype=float)
    route_rrp = np.asarray(trace.get("rrp_route_mean", []), dtype=float)
    route_replenishment_resource = np.asarray(
        trace.get("replenishment_resource_route_mean", []), dtype=float
    )

    result = {
        "frequency_hz": float(np.mean(frequencies)) if frequencies else float("nan"),
        "frequency_sd_population_hz": float(np.std(frequencies, ddof=1)) if len(frequencies) > 1 else float("nan"),
        "lr_phase_error_mean_abs_deg": float(np.mean(np.abs(lr_errors))) if len(lr_errors) else float("nan"),
        "lr_phase_error_sd_deg": float(np.std(lr_errors, ddof=1)) if len(lr_errors) > 1 else float("nan"),
        "fe_phase_error_mean_abs_deg": float(np.mean(np.abs(fe_errors))) if len(fe_errors) else float("nan"),
        "fe_phase_error_sd_deg": float(np.std(fe_errors, ddof=1)) if len(fe_errors) > 1 else float("nan"),
        "post_pulse_lr_phase_error_mean_abs_deg": float(np.mean(np.abs(post_lr))) if len(post_lr) else float("nan"),
        "post_pulse_fe_phase_error_mean_abs_deg": float(np.mean(np.abs(post_fe))) if len(post_fe) else float("nan"),
        "sham_excitatory_lr_phase_error_mean_abs_deg": float(np.mean(np.abs(sham_excitatory_lr))) if len(sham_excitatory_lr) else float("nan"),
        "sham_excitatory_fe_phase_error_mean_abs_deg": float(np.mean(np.abs(sham_excitatory_fe))) if len(sham_excitatory_fe) else float("nan"),
        "sham_inhibitory_lr_phase_error_mean_abs_deg": float(np.mean(np.abs(sham_inhibitory_lr))) if len(sham_inhibitory_lr) else float("nan"),
        "sham_inhibitory_fe_phase_error_mean_abs_deg": float(np.mean(np.abs(sham_inhibitory_fe))) if len(sham_inhibitory_fe) else float("nan"),
        "lr_phase_slip_rate": float(np.mean(np.abs(lr_errors) > cfg.phase_slip_threshold_deg)) if len(lr_errors) else float("nan"),
        "fe_phase_slip_rate": float(np.mean(np.abs(fe_errors) > cfg.phase_slip_threshold_deg)) if len(fe_errors) else float("nan"),
        "post_pulse_lr_phase_slip_rate": float(np.mean(np.abs(post_lr) > cfg.phase_slip_threshold_deg)) if len(post_lr) else float("nan"),
        "post_pulse_fe_phase_slip_rate": float(np.mean(np.abs(post_fe) > cfg.phase_slip_threshold_deg)) if len(post_fe) else float("nan"),
        "lr_phase_slip_count": int(np.sum(np.abs(lr_errors) > cfg.phase_slip_threshold_deg)),
        "lr_phase_cycle_count": int(len(lr_errors)),
        "fe_phase_slip_count": int(np.sum(np.abs(fe_errors) > cfg.phase_slip_threshold_deg)),
        "fe_phase_cycle_count": int(len(fe_errors)),
        "post_pulse_lr_phase_slip_count": int(np.sum(np.abs(post_lr) > cfg.phase_slip_threshold_deg)),
        "post_pulse_lr_phase_cycle_count": int(len(post_lr)),
        "post_pulse_fe_phase_slip_count": int(np.sum(np.abs(post_fe) > cfg.phase_slip_threshold_deg)),
        "post_pulse_fe_phase_cycle_count": int(len(post_fe)),
        "sham_excitatory_lr_phase_slip_count": int(np.sum(np.abs(sham_excitatory_lr) > cfg.phase_slip_threshold_deg)),
        "sham_excitatory_lr_phase_cycle_count": int(len(sham_excitatory_lr)),
        "sham_excitatory_fe_phase_slip_count": int(np.sum(np.abs(sham_excitatory_fe) > cfg.phase_slip_threshold_deg)),
        "sham_excitatory_fe_phase_cycle_count": int(len(sham_excitatory_fe)),
        "sham_inhibitory_lr_phase_slip_count": int(np.sum(np.abs(sham_inhibitory_lr) > cfg.phase_slip_threshold_deg)),
        "sham_inhibitory_lr_phase_cycle_count": int(len(sham_inhibitory_lr)),
        "sham_inhibitory_fe_phase_slip_count": int(np.sum(np.abs(sham_inhibitory_fe) > cfg.phase_slip_threshold_deg)),
        "sham_inhibitory_fe_phase_cycle_count": int(len(sham_inhibitory_fe)),
        "rg_cycle_interval_cv_mean": float(np.mean(cycle_cvs)) if len(cycle_cvs) else float("nan"),
        "bilateral_amplitude_balance": float(1.0 - imbalance) if np.isfinite(imbalance) else float("nan"),
        "bilateral_amplitude_imbalance": float(imbalance),
        "pf_transfer_reliability": float(1.0 - total_pf_missed / total_rg_pf) if total_rg_pf else float("nan"),
        "mn_transfer_reliability": float(1.0 - total_mn_missed / total_rg_mn) if total_rg_mn else float("nan"),
        "pf_transfer_anchor_count": int(total_rg_pf),
        "pf_transfer_missed_count": int(total_pf_missed),
        "pf_transfer_matched_count": int(total_rg_pf - total_pf_missed),
        "mn_transfer_anchor_count": int(total_rg_mn),
        "mn_transfer_missed_count": int(total_mn_missed),
        "mn_transfer_matched_count": int(total_rg_mn - total_mn_missed),
        "pf_missed_burst_fraction": float(total_pf_missed / total_rg_pf) if total_rg_pf else float("nan"),
        "mn_missed_burst_fraction": float(total_mn_missed / total_rg_mn) if total_rg_mn else float("nan"),
        "rg_pf_latency_mean_ms": float(1000.0 * np.mean(pf_latency_array)) if len(pf_latency_array) else float("nan"),
        "rg_mn_latency_mean_ms": float(1000.0 * np.mean(mn_latency_array)) if len(mn_latency_array) else float("nan"),
        **recovery_outcome,
        "pulse_start_s": pulse_start,
        "pulse_end_s": pulse_end,
        "sham_excitatory_start_s": sham_excitatory_start,
        "sham_inhibitory_start_s": sham_inhibitory_start,
        "pulse_required": int(pulse_required),
        "pulse_delivered": pulse_delivered,
        "pulse_response_eligible": pulse_response_eligible,
        "pulse_noneligibility_reason": pulse_noneligibility_reason,
        "recovery_composite_eligible": recovery_composite_eligible,
        "recovery_composite_event": recovery_composite_event,
        "recovery_composite_time_s": recovery_composite_time_s,
        "baseline_cycles_before_pulse_min": baseline_cycles_before_pulse,
        "baseline_cycles_before_excitatory_sham_min": baseline_cycles_before_excitatory_sham,
        "baseline_cycles_before_inhibitory_sham_min": baseline_cycles_before_inhibitory_sham,
        "sham_excitatory_endpoint_eligible": int(
            bool(quality["technical_valid"])
            and baseline_cycles_before_excitatory_sham >= 2
        ),
        "sham_inhibitory_endpoint_eligible": int(
            bool(quality["technical_valid"])
            and baseline_cycles_before_inhibitory_sham >= 2
        ),
        "rhythmic_failure": rhythmic_failure,
        **quality,
        "ia_signal_mean": float(np.mean(trace["ia_signal"][time_mask])),
        "ib_signal_mean": float(np.mean(trace["ib_signal"][time_mask])),
        "muscle_force_mean": float(np.mean(trace["muscle_force"][time_mask])),
        "pre_transition_frequency_hz": pre_frequency,
        "post_transition_frequency_hz": post_frequency,
        "post_pre_frequency_ratio": float(post_frequency / pre_frequency) if pre_frequency > 0 else float("nan"),
        "mean_mt_support_left": float(np.mean(trace["mt_side"][time_mask, 0])),
        "mean_mt_support_right": float(np.mean(trace["mt_side"][time_mask, 1])),
        **relay_rates,
        **burst_counts,
    }
    # Mediator outputs are retained for mechanism checks only; primary analysis
    # is statically forbidden from reading them.
    if (
        route_support.ndim == 2
        and route_rrp.ndim == 2
        and route_replenishment_resource.ndim == 2
    ):
        for index, route in enumerate(MT_ROUTES):
            result[f"mt_{route}_mean"] = float(np.mean(route_support[time_mask, index]))
            result[f"rrp_{route}_mean"] = float(np.mean(route_rrp[time_mask, index]))
            result[f"replenishment_resource_{route}_mean"] = float(np.mean(
                route_replenishment_resource[time_mask, index]
            ))
    # Backward-compatible aliases retained only for old audit scripts.
    result["left_right_phase_error_mean_abs_deg"] = result["lr_phase_error_mean_abs_deg"]
    result["global_phase_error_mean_abs_deg"] = float(np.mean(np.abs(_safe_concat((lr_errors, fe_errors))))) if len(lr_errors) + len(fe_errors) else float("nan")
    result["post_perturbation_phase_slip_fraction"] = float(np.mean(np.abs(_safe_concat((post_lr, post_fe))) > cfg.phase_slip_threshold_deg)) if len(post_lr) + len(post_fe) else float("nan")
    result["alternation_index"] = 1.0 - min(result["lr_phase_error_mean_abs_deg"] / 180.0, 1.0) if np.isfinite(result["lr_phase_error_mean_abs_deg"]) else float("nan")
    return result


def analysis_event_payload(
    trace: Mapping[str, np.ndarray], cfg: Config
) -> Dict[str, object]:
    """Compact, lossless burst events needed to recompute primary endpoints.

    Full millisecond traces are prohibitively large for the complete A-H
    matrix. Persisting all RG/PF/MN burst onsets preserves the raw event-level
    information behind phase, transfer and recovery metrics, providing a
    re-analysis path without rerunning the expensive simulations.
    """
    payload: Dict[str, object] = {}
    for cell_class in ("RG", "PF", "MN"):
        for name, events in detect_population_bursts(
            trace, cfg, cell_class
        ).items():
            payload[f"{name}_onset_s"] = events.tolist()
    return payload


def analysis_observable_payload(
    trace: Mapping[str, np.ndarray], cfg: Config
) -> Dict[str, object]:
    """Exact sufficient observables for the bilateral MN-amplitude endpoint.

    Persisting the two sums and their common sample count lets preflight
    independently reconstruct the Family-6 left/right mean MN population-rate
    amplitude without trusting the derived summary scalar or storing the full
    millisecond trace.
    """
    time_s = np.asarray(trace["time_s"], dtype=float)
    rate_hz = np.asarray(trace["rate_hz"], dtype=float)
    mask = (time_s >= cfg.burn_in_s) & (time_s <= cfg.duration_s)
    sample_count = int(np.sum(mask))
    if sample_count <= 0:
        raise ValueError("analysis-observable interval has no MN samples")
    left_indices = [pop("MN", "L", phase) for phase in PHASES]
    right_indices = [pop("MN", "R", phase) for phase in PHASES]
    left_sum = float(np.sum(rate_hz[mask][:, left_indices], dtype=np.float64))
    right_sum = float(np.sum(rate_hz[mask][:, right_indices], dtype=np.float64))
    if not (math.isfinite(left_sum) and math.isfinite(right_sum)):
        raise ValueError("analysis-observable MN sums must be finite")
    if left_sum < 0.0 or right_sum < 0.0:
        raise ValueError("analysis-observable MN sums must be nonnegative")
    return {
        "mn_left_rate_sum_hz_samples": left_sum,
        "mn_right_rate_sum_hz_samples": right_sum,
        "mn_rate_sample_count": sample_count,
    }


def intervention_log_payload(
    trace: Mapping[str, np.ndarray], cfg: Config
) -> Dict[str, object]:
    """Raw pulse-delivery log used to reconstruct Family-9 eligibility.

    This is an intervention record, not a recovery outcome.  Burst-based
    baseline eligibility and recovery are recomputed from ``analysis_events``.
    A phase-triggered pulse that could not be scheduled is retained as the
    preregistered biological non-delivery state rather than silently excluded.
    """
    protocol = str(np.asarray(trace["protocol"])[0])
    direction = str(np.asarray(trace["pulse_direction"])[0])
    pulse_required = int(protocol == "pulse" and direction != "none")
    start = float(np.asarray(trace.get("pulse_start_s", [np.nan]))[0])
    end = float(np.asarray(trace.get("pulse_end_s", [np.nan]))[0])
    pulse_delivered = int(
        pulse_required
        and math.isfinite(start)
        and math.isfinite(end)
        and 0.0 <= start < end <= cfg.duration_s
    )
    if pulse_delivered:
        reason = "none"
    elif pulse_required:
        reason = "biological_no_phase_eligible_cycle"
    else:
        reason = "no_pulse_condition"
    return {
        "pulse_required": pulse_required,
        "pulse_delivered": pulse_delivered,
        "pulse_start_s": start,
        "pulse_end_s": end,
        "pulse_noneligibility_reason": reason,
    }


def summarize_long_epochs(
    trace: Mapping[str, np.ndarray], cfg: Config
) -> List[Dict[str, object]]:
    """Epoch-resolved demand/challenge/recovery observations.

    The long protocol keeps left-right coordination, flexor-extensor
    coordination, rhythm regularity, bilateral motor amplitude and transfer
    observations separate. MT/RRP/slow-resource values are secondary mediators and
    never substitute for a functional endpoint.
    """
    if str(trace["protocol"][0]) != "long":
        raise ValueError("summarize_long_epochs requires protocol='long'")
    rg_bursts = detect_population_bursts(trace, cfg, "RG")
    pf_bursts = detect_population_bursts(trace, cfg, "PF")
    mn_bursts = detect_population_bursts(trace, cfg, "MN")
    quality = technical_trace_quality(trace)
    rows: List[Dict[str, object]] = []
    for epoch in range(1, cfg.long_n_epochs + 1):
        start = (epoch - 1) * cfg.long_epoch_duration_s
        end = epoch * cfg.long_epoch_duration_s
        if epoch <= cfg.long_baseline_end_epoch:
            stage = "baseline"
        elif epoch < cfg.long_challenge_epoch:
            stage = "prechallenge_demand"
        elif epoch <= cfg.long_demand_end_epoch:
            stage = "challenged_demand"
        else:
            stage = "postdemand_recovery"

        selected: Dict[str, np.ndarray] = {}
        counts: List[int] = []
        cvs: List[float] = []
        freqs: List[float] = []
        for side, phase in SIDE_PHASES:
            name = population_name("RG", side, phase)
            events = rg_bursts[name]
            selected[name] = events[(events >= start) & (events < end)]
            counts.append(len(selected[name]))
            intervals = np.diff(selected[name])
            if len(intervals) >= 2 and np.mean(intervals) > 0:
                cvs.append(float(np.std(intervals, ddof=1) / np.mean(intervals)))
                freqs.append(float(1.0 / np.mean(intervals)))

        lr_errors = _safe_concat((
            cycle_phase_errors_deg(
                selected[population_name("RG", "L", "F")],
                rg_bursts[population_name("RG", "R", "F")], start,
            ),
            cycle_phase_errors_deg(
                selected[population_name("RG", "L", "E")],
                rg_bursts[population_name("RG", "R", "E")], start,
            ),
        ))
        fe_errors = _safe_concat((
            cycle_phase_errors_deg(
                selected[population_name("RG", "L", "F")],
                rg_bursts[population_name("RG", "L", "E")], start,
            ),
            cycle_phase_errors_deg(
                selected[population_name("RG", "R", "F")],
                rg_bursts[population_name("RG", "R", "E")], start,
            ),
        ))
        # Sampled states are labeled by their integration interval's common
        # right endpoint, so epoch state summaries use (start, end]. Discrete
        # burst events above retain the conventional physical-time [start, end).
        mask = right_endpoint_sample_mask(trace["time_s"], start, end)
        mn_left = [pop("MN", "L", phase) for phase in PHASES]
        mn_right = [pop("MN", "R", phase) for phase in PHASES]
        left_amplitude = float(np.mean(np.sum(trace["rate_hz"][mask][:, mn_left], axis=1)))
        right_amplitude = float(np.mean(np.sum(trace["rate_hz"][mask][:, mn_right], axis=1)))
        amplitude_sum = left_amplitude + right_amplitude
        balance = (
            1.0 - abs(left_amplitude - right_amplitude) / amplitude_sum
            if amplitude_sum > 0 else float("nan")
        )

        pf_total = pf_missed = mn_total = mn_missed = 0
        for side, phase in SIDE_PHASES:
            rg_name = population_name("RG", side, phase)
            pf_name = population_name("PF", side, phase)
            mn_name = population_name("MN", side, phase)
            _, missed, total = match_output_bursts(
                rg_bursts[rg_name], pf_bursts[pf_name], start, end,
                0.0, cfg.rg_pf_match_window_s,
            )
            pf_missed += missed
            pf_total += total
            _, missed, total = match_output_bursts(
                rg_bursts[rg_name], mn_bursts[mn_name], start, end,
                cfg.rg_mn_match_pre_window_s, cfg.rg_mn_match_post_window_s,
            )
            mn_missed += missed
            mn_total += total

        row: Dict[str, object] = {
            "epoch": epoch,
            "stage": stage,
            "frequency_hz": float(np.mean(freqs)) if freqs else float("nan"),
            "rg_cycle_interval_cv_mean": float(np.mean(cvs)) if cvs else float("nan"),
            "lr_phase_error_mean_abs_deg": float(np.mean(np.abs(lr_errors))) if len(lr_errors) else float("nan"),
            "fe_phase_error_mean_abs_deg": float(np.mean(np.abs(fe_errors))) if len(fe_errors) else float("nan"),
            "lr_phase_slip_rate": float(np.mean(np.abs(lr_errors) > cfg.phase_slip_threshold_deg)) if len(lr_errors) else float("nan"),
            "fe_phase_slip_rate": float(np.mean(np.abs(fe_errors) > cfg.phase_slip_threshold_deg)) if len(fe_errors) else float("nan"),
            "lr_phase_slip_count": int(np.sum(
                np.abs(lr_errors) > cfg.phase_slip_threshold_deg
            )),
            "lr_phase_cycle_count": int(len(lr_errors)),
            "fe_phase_slip_count": int(np.sum(
                np.abs(fe_errors) > cfg.phase_slip_threshold_deg
            )),
            "fe_phase_cycle_count": int(len(fe_errors)),
            "bilateral_amplitude_balance": balance,
            "pf_transfer_reliability": 1.0 - pf_missed / pf_total if pf_total else float("nan"),
            "mn_transfer_reliability": 1.0 - mn_missed / mn_total if mn_total else float("nan"),
            "pf_transfer_anchor_count": int(pf_total),
            "pf_transfer_missed_count": int(pf_missed),
            "pf_transfer_matched_count": int(pf_total - pf_missed),
            "mn_transfer_anchor_count": int(mn_total),
            "mn_transfer_missed_count": int(mn_missed),
            "mn_transfer_matched_count": int(mn_total - mn_missed),
            "rhythmic_failure": int(min(counts) < 2),
            "min_rg_burst_count": int(min(counts)),
            "challenged_rrp_mean_secondary": float(np.mean(trace["challenged_rrp"][mask])),
            "challenged_replenishment_resource_mean_secondary": float(
                np.mean(trace["challenged_replenishment_resource"][mask])
            ),
            **quality,
        }
        for (side, phase), count in zip(SIDE_PHASES, counts):
            row[f"n_bursts_{side}_{phase}"] = int(count)
        for route_index, route in enumerate(MT_ROUTES):
            row[f"mt_{route}_mean"] = float(np.mean(
                trace["mt_route_support"][mask, route_index]
            ))
            row[f"rrp_{route}_mean_secondary"] = float(np.mean(
                trace["rrp_route_mean"][mask, route_index]
            ))
            row[f"replenishment_resource_{route}_mean_secondary"] = float(np.mean(
                trace["replenishment_resource_route_mean"][mask, route_index]
            ))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def connection_rows(cfg: Config, trace: Mapping[str, np.ndarray]) -> List[Dict[str, object]]:
    specs = pathway_specs(cfg)
    sizes = trace["population_sizes"]
    pathway_index = trace["connectome_pathway_index"]
    delays = trace["connectome_delay_ms"]
    weights = trace["connectome_weight_pa"]
    rows: List[Dict[str, object]] = []
    for index, spec in enumerate(specs):
        mask = pathway_index == index
        possible = int(sizes[spec.source_population] * sizes[spec.target_population])
        if spec.source_population == spec.target_population:
            possible -= int(sizes[spec.source_population])
        rows.append({
            "pathway": spec.name,
            "source": POPULATIONS[spec.source_population],
            "target": POPULATIONS[spec.target_population],
            "sign": "excitatory" if spec.population_weight_pa > 0 else "inhibitory",
            "functional_role": spec.functional_role,
            "mt_route": spec.mt_route,
            "recruitment_axis": spec.recruitment_axis,
            "evidence_class": spec.evidence_class,
            "evidence_note": spec.evidence_note,
            "mt_resource_dependent": spec.mt_route != "none",
            "mt_action": (
                "MT_modulated_slow_RRP_replenishment_with_normalized_resource"
                if spec.mt_route != "none" else "mt_independent"
            ),
            "specified_connection_probability": spec.connection_probability,
            "realized_connection_probability": float(np.sum(mask) / possible),
            "edge_count": int(np.sum(mask)),
            "mean_indegree": float(np.sum(mask) / sizes[spec.target_population]),
            "population_weight_pa": spec.population_weight_pa,
            "per_edge_weight_pa": float(np.mean(weights[mask])),
            "mean_delay_ms": float(np.mean(delays[mask])),
            "min_delay_ms": float(np.min(delays[mask])),
            "max_delay_ms": float(np.max(delays[mask])),
            "parameter_status": "modeling_prior_for_sensitivity_analysis",
        })
    return rows


def plot_network_trace(trace: Mapping[str, np.ndarray], cfg: Config, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    colors = {
        "RG": "#2563eb", "PF": "#0891b2", "MN": "#64748b",
        "V0D": "#ef4444", "V0V": "#f59e0b", "V3": "#16a34a",
        "V2a": "#7c3aed", "V1Ia": "#dc2626", "V1Ren": "#b91c1c",
        "V2b": "#be123c",
    }
    spike_t = trace["spike_time_s"]
    spike_p = trace["spike_population"]
    spike_n = trace["spike_neuron"]
    offsets = np.cumsum(np.r_[0, trace["population_sizes"][:-1]])
    for p_index, name in enumerate(POPULATIONS):
        cell_class = name.rsplit("_", 2)[0]
        mask = spike_p == p_index
        axes[0].scatter(spike_t[mask], offsets[p_index] + spike_n[mask], s=0.7, color=colors[cell_class], rasterized=True)
    for side, phase, color in (("L", "F", "#2563eb"), ("L", "E", "#60a5fa"), ("R", "F", "#dc2626"), ("R", "E", "#f87171")):
        p_index = pop("RG", side, phase)
        axes[1].plot(trace["time_s"], trace["rate_hz"][:, p_index], lw=1.0, color=color, label=f"RG-{side}{phase}")
        pf_index = pop("PF", side, phase)
        axes[2].plot(trace["time_s"], trace["rate_hz"][:, pf_index], lw=1.0, color=color, label=f"PF-{side}{phase}")
    for route_index, route in enumerate(MT_ROUTES):
        color = colors[route]
        axes[3].plot(
            trace["time_s"], trace["mt_route_support"][:, route_index],
            color=color, lw=0.9, label=f"M-{route}",
        )
    for index, (side, phase) in enumerate(SIDE_PHASES):
        color = ("#2563eb", "#60a5fa", "#dc2626", "#f87171")[index]
        axes[4].plot(trace["time_s"], trace["mn_hz"][:, index], color=color, lw=1.0, label=f"MN-{side}{phase}")
    axes[0].set_ylabel("Spiking cells")
    axes[1].set_ylabel("RG rate (Hz)")
    axes[2].set_ylabel("PF rate (Hz)")
    axes[3].set_ylabel("MT / resource")
    axes[4].set_ylabel("MN readout")
    axes[4].set_xlabel("Time (s)")
    axes[1].legend(ncol=4, frameon=False)
    axes[2].legend(ncol=4, frameon=False)
    axes[3].legend(ncol=3, frameon=False)
    protocol = str(trace["protocol"][0])
    if protocol in {"noise_burst", "pf_deletion", "phase_kick", "pulse"}:
        if protocol == "pulse":
            start = float(trace["pulse_start_s"][0])
            end = float(trace["pulse_end_s"][0])
        else:
            start, end = cfg.perturbation_start_s, cfg.perturbation_end_s
        for axis in axes:
            axis.axvspan(
                start, end,
                color="#f59e0b", alpha=0.10, lw=0,
            )
        label = {
            "noise_burst": "noise burst",
            "pf_deletion": "PF-LF deletion",
            "phase_kick": "unilateral RG phase kick",
            "pulse": f"{trace['pulse_direction'][0]} pulse",
        }[protocol]
        axes[0].text(
            (start + end) / 2.0,
            1.01, label, transform=axes[0].get_xaxis_transform(),
            ha="center", va="bottom", color="#92400e", fontsize=9,
        )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    default_cfg = Config()
    duration_s = args.duration
    if duration_s is None:
        duration_s = (
            default_cfg.long_n_epochs * default_cfg.long_epoch_duration_s
            if args.protocol == "long" else default_cfg.duration_s
        )
    cfg = Config(
        duration_s=duration_s,
        dt_ms=default_cfg.dt_ms if args.dt_ms is None else args.dt_ms,
        rg_neurons=args.rg_neurons,
        pf_neurons=args.pf_neurons,
        relay_neurons=args.relay_neurons,
        mn_neurons=args.mn_neurons,
    )
    outdir = Path(args.outdir).expanduser().resolve()
    (outdir / "data").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    trace = simulate(
        cfg, args.seed, args.protocol, args.mt_mode,
        structural_seed=args.structural_seed,
        static_scale=args.static_scale,
        speed_level=args.speed,
        load_context=args.load,
        load_side=args.load_side,
        pulse_direction=args.pulse_direction,
        pulse_cycle_fraction_override=args.pulse_cycle_fraction,
        pulse_target_side=args.pulse_target_side,
        pulse_target_phase=args.pulse_target_phase,
        ablated_populations=args.ablate,
        impaired_mt_routes=args.impaired_mt_route,
        challenged_routes=args.challenged_route or MT_ROUTES,
        fast_mode=args.fast_mode,
    )
    summary = summarize(trace, cfg)
    np.savez_compressed(outdir / "data" / "explicit_network_trace.npz", **trace)
    write_csv(outdir / "data" / "cellular_summary.csv", [{"seed": args.seed, **summary}])
    write_csv(outdir / "data" / "architecture_connections.csv", connection_rows(cfg, trace))
    write_simulation_manifest(outdir / "data" / "simulation_manifest.json", trace, cfg)
    with (outdir / "data" / "settings.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": asdict(cfg), "populations": POPULATIONS, "arguments": vars(args)}, handle, indent=2)
    plot_network_trace(trace, cfg, outdir / "figures" / "explicit_network_trace.png")
    print(json.dumps(json_safe({"summary": summary}), indent=2, allow_nan=False))
    print(f"Outputs written to: {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="explicit_commissural_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protocol",
        choices=("steady_state", "pulse", "long", "noise_burst", "speed_step", "phase_kick", "pf_deletion"),
        default="steady_state",
    )
    parser.add_argument(
        "--mt-mode",
        choices=MT_MODES,
        default="dynamic",
    )
    parser.add_argument("--structural-seed", type=int, default=None)
    parser.add_argument("--static-scale", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--dt-ms", type=float, default=None)
    parser.add_argument("--rg-neurons", type=int, default=16)
    parser.add_argument("--pf-neurons", type=int, default=12)
    parser.add_argument("--relay-neurons", type=int, default=8)
    parser.add_argument("--mn-neurons", type=int, default=12)
    parser.add_argument(
        "--fast-mode", choices=("dynamic", "static_mean", "off"),
        default="dynamic",
    )
    parser.add_argument("--speed", choices=SPEED_LEVELS, default="medium")
    parser.add_argument("--load", choices=LOAD_CONTEXTS, default="normal")
    parser.add_argument("--load-side", choices=SIDES, default="L")
    parser.add_argument("--pulse-direction", choices=PULSE_DIRECTIONS, default="none")
    parser.add_argument("--pulse-cycle-fraction", type=float, default=None)
    parser.add_argument("--pulse-target-side", choices=SIDES, default="R")
    parser.add_argument("--pulse-target-phase", choices=PHASES, default="F")
    parser.add_argument(
        "--ablate", action="append", default=[],
        choices=(*CLASSES, *AFFERENT_ABLATIONS),
    )
    parser.add_argument("--impaired-mt-route", action="append", default=[], choices=MT_ROUTES)
    parser.add_argument("--challenged-route", action="append", default=[], choices=MT_ROUTES)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
