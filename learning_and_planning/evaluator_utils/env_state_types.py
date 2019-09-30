from enum import Enum


class EnvState(Enum):
  # The mapping to integers IS important and used in many places.
  DEAD = 0  # Dead-end, level cannot be solved.
  SOLVED = 1  # Level is solved.
  SOLVABLE = 2  # Level is neither solved or in dead-end.
