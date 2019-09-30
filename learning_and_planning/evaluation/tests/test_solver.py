from learning_and_planning.evaluation.examples import TrivialSolver
from learning_and_planning.evaluation.tests.utils import MockEnv


class TestTrivialSolver:
    solver = TrivialSolver()

    def test_trivial_solver_solves_trivial_level(self):
        env = MockEnv([2])
        env.reset()
        solved, num_steps = self.solver.solve(env)
        assert solved
        assert num_steps == 2

    def test_budget_limits_number_of_steps(self):
        env = MockEnv([15])
        env.reset()
        solved, num_steps = self.solver.solve(env, budget=10)
        assert not solved
        assert num_steps == 10

    def test_limited_budget_prevents_from_solving_level(self):
        env = MockEnv([2])
        env.reset()
        solved, num_steps = self.solver.solve(env)
        assert solved
        assert num_steps == 2

        env = MockEnv([2])
        env.reset()
        solved, num_steps = self.solver.solve(env, budget=1)
        assert not solved
        assert num_steps == 1
