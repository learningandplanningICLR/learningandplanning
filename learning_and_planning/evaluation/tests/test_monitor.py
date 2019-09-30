import pytest

from gym_sokoban.envs import SokobanEnv

from learning_and_planning.evaluation import monitor
from learning_and_planning.evaluation.examples import RandomSolver
from learning_and_planning.evaluation.metrics import Evaluator
from learning_and_planning.utils import wrappers


def test_video_recording():
    solver = RandomSolver()
    env = wrappers.InfoDisplayWrapper(
        wrappers.RewardPrinter(
            SokobanEnv(
                dim_room=(8, 8),
                num_boxes=1,
                max_steps=50,
            )
        ),
        augment_observations=True,
        min_text_area_width=500
    )
    Evaluator(env, None, 10, video_directory='test_monitor', log_interval=1).evaluate(solver)


# Skip during tests, because requires interaction with user
@pytest.mark.skip
def test_recording_play():
    env = wrappers.PlayWrapper(
        monitor.EvaluationMonitor(
            wrappers.InfoDisplayWrapper(
                wrappers.RewardPrinter(
                    SokobanEnv()
                ),
                augment_observations=True,
                min_text_area_width=500
            ),
            directory='test_recording_play',
            force=True,
            video_callable=lambda *args: True
        )
    )
    env.play()
