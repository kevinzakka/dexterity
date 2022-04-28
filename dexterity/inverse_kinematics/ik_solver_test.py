"""Tests for ik_solver."""

import numpy as np
from absl.testing import absltest

from dexterity.inverse_kinematics import ik_solver
from dexterity.models.hands import shadow_hand_e


class IKSolverTest(absltest.TestCase):
    def test_return_none_when_passing_impossible_target(self) -> None:
        hand = shadow_hand_e.ShadowHandSeriesE()
        target_positions = np.full(shape=(5, 3), fill_value=10.0)
        solver = ik_solver.IKSolver(hand)
        qpos_sol = solver.solve(target_positions)
        self.assertIsNone(qpos_sol)


if __name__ == "__main__":
    absltest.main()
