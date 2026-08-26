#!/usr/bin/env python3
"""Contract tests for the v2.6.2 single-realization execution overlay."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import analyze_single_realization_v2_6_2 as analysis
import distributed_single_realization_v2_6_2 as distributed
import run_single_realization_v2_6_2 as single


class SingleRealizationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.tasks,
            cls.representatives,
            cls.task_to_sim,
            cls.multiplicity,
        ) = single.build_matrix()

    def test_exact_frozen_matrix(self) -> None:
        actual = single.assert_matrix_contract(self.tasks, self.representatives)
        self.assertEqual(actual, single.expected_matrix_contract())

    def test_only_seed_601_is_materialized(self) -> None:
        self.assertEqual({task.seed for task in self.tasks}, {601})
        identities = [
            single.accelerated.simulation_identity(task, smoke=False)
            for task in self.representatives.values()
        ]
        self.assertEqual({item["seed"] for item in identities}, {601})
        self.assertEqual({item["structural_seed"] for item in identities}, {160601})

    def test_all_biological_factorial_axes_are_retained(self) -> None:
        stage_f = [task for task in self.tasks if task.stage == "F"]
        self.assertEqual(len(stage_f), 10 * 10 * 4 * 27)
        self.assertEqual(
            {(task.ablations[0] if task.ablations else None) for task in stage_f},
            {None, *single.model.CLASSES},
        )
        self.assertEqual(
            {(task.impaired_mt_routes[0] if task.impaired_mt_routes else None)
             for task in stage_f},
            {None, *single.model.MT_ROUTES},
        )

    def test_release_contract_is_exact(self) -> None:
        release = single.assert_release_contract()
        self.assertFalse(release["model_equations_or_parameters_changed"])
        self.assertFalse(release["stochastic_population_inference_authorized"])

    def test_plan_is_resume_locked(self) -> None:
        matrix = single.assert_matrix_contract(self.tasks, self.representatives)
        plan = single.execution_plan(matrix)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            single.write_or_verify_plan(output, plan)
            single.write_or_verify_plan(output, plan)
            changed = dict(plan)
            changed["fixed_seed"] = 999
            with self.assertRaises(RuntimeError):
                single.write_or_verify_plan(output, changed)

    def test_descriptive_result_has_no_inferential_statistics(self) -> None:
        row = analysis.contrast_record(1, -2.0)
        forbidden = {
            "p_value", "p_two_sided", "alpha", "standard_error", "ci95",
            "t_statistic", "degrees_freedom", "paired_effect_dz",
        }
        self.assertTrue(forbidden.isdisjoint(row))
        self.assertEqual(row["independent_stochastic_realization_count"], 1)
        self.assertTrue(row["direction_is_favorable"])

    def test_three_shards_cover_every_simulation_once(self) -> None:
        manifest = distributed.build_manifest(3, "/opt/cpg/production")
        rows = manifest["assignments"]
        self.assertEqual(len(rows), single.EXPECTED_UNIQUE_SIMULATION_COUNT)
        self.assertEqual(
            len({row["simulation_id"] for row in rows}), len(rows)
        )
        self.assertEqual(sum(manifest["shard_unique_simulation_counts"]), len(rows))
        self.assertEqual(set(manifest["shard_unique_simulation_counts"]), {1203, 1204})


if __name__ == "__main__":
    unittest.main()
