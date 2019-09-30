from unittest import mock
from unittest.mock import call, MagicMock

from learning_and_planning.evaluation.examples import TrivialSolver
from learning_and_planning.evaluation.metrics import Evaluator
from learning_and_planning.evaluation.tests.utils import MockEnv


@mock.patch('learning_and_planning.evaluation.metrics.SokobanEnv')
def test_evaluation_for_trivial_solver(EnvClass):
    env = MockEnv(
        [3, 4, 5, 6, 7],
        [True, True, False, True, False]
    )
    solver = TrivialSolver()
    solved_ratio = Evaluator(env, None, 5, None).evaluate(solver)
    assert solved_ratio == 0.6
    EnvClass.assert_not_called()

    EnvClass.return_value = MockEnv(
        [3, 4, 5, 6, 7],
        [True, True, False, True, False]
    )
    solved_ratio = Evaluator(None, None, 5, None).evaluate(solver)
    assert solved_ratio == 0.6


class TestParametrizedSolver:
    solve_returns = [
        (False, 50),
        (True, 6),
        (False, 50),
        (True, 100)
    ]

    def test_number_of_levels_used(self):
        solver = mock.MagicMock()
        solver.solve.side_effect = self.solve_returns
        env = MagicMock()
        Evaluator(env, None, 3).evaluate(solver)
        solver.solve.assert_has_calls([call(env, None)] * 3)

    def test_left_budget_is_calculated(self):
        solver = mock.MagicMock()
        solver.solve.side_effect = self.solve_returns
        env = MagicMock()
        total_budget = 200
        Evaluator(env, None, 3, total_budget=total_budget).evaluate(solver)
        solver.solve.assert_has_calls([
            call(env, total_budget),
            call(env, total_budget - 50),
            call(env, total_budget - 56)
        ])

    def test_no_budget_left_for_all_levels(self):
        solver = mock.MagicMock()
        solver.solve.side_effect = self.solve_returns
        env = MagicMock()
        total_budget = 100
        Evaluator(env, None, 10, total_budget=total_budget).evaluate(solver)
        solver.solve.assert_has_calls([
            call(env, total_budget),
            call(env, total_budget - 50),
            call(env, total_budget - 56),
        ])
