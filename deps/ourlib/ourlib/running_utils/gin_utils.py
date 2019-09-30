import tempfile

import gin
from pathlib2 import Path


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in
      the config files.
  """
  gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)

def load_gin_config(gin_config):
        gin_file_path = Path(tempfile.mktemp())
        gin_file_path.write_text(gin_config)
        load_gin_configs([str(gin_file_path)], gin_bindings=[])



