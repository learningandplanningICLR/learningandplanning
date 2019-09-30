import pickle
from pathlib import Path


class DataReader(object):
  def __init__(self, data_files_prefix):
    self.number_of_shards = 0
    while Path(data_files_prefix + "_{:04d}".format(self.number_of_shards)).is_file():
      self.number_of_shards += 1
    print("Detected {} shards".format(self.number_of_shards))

    self.data_files_prefix = data_files_prefix
    self.preprocess_type = "vf"

  def _load_shard(self, shard):
    if self.preprocess_type=="vf":
      return self._load_shard_vf(shard)

  def _load_shard_vf(self, shard):
    file_name = Path(self.data_files_prefix + "_{:04d}".format(shard))
    with open(file_name, "rb") as file:
      data = pickle.load(file)
    return data

  def load(self):
    unpckd_all = []
    for shard in range(self.number_of_shards):
      unpckd_all.extend(self._load_shard(shard))
    return unpckd_all

