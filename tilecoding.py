#!/usr/bin/env python
import numpy as np

class TileCoder:
  def __init__(self, tiles_per_dim, value_limits, tilings, offset=lambda n: 2 * np.arange(n) + 1):
    tiling_dims = np.array(np.ceil(tiles_per_dim), dtype=np.int) + 1
    self._offsets = offset(len(tiles_per_dim)) * \
      np.repeat([np.arange(tilings)], len(tiles_per_dim), 0).T / float(tilings) % 1
    self._limits = np.array(value_limits)
    self._norm_dims = np.array(tiles_per_dim) / (self._limits[:, 1] - self._limits[:, 0])
    self._tile_base_ind = np.prod(tiling_dims) * np.arange(tilings)
    self._hash_vec = np.array([np.prod(tiling_dims[0:i]) for i in range(len(tiles_per_dim))])
    self._n_tiles = tilings * np.prod(tiling_dims)
  
  def __getitem__(self, x):
    off_coords = ((x - self._limits[:, 0]) * self._norm_dims + self._offsets).astype(int)
    return self._tile_base_ind + np.dot(off_coords, self._hash_vec)

  @property
  def n_tiles(self):
    return self._n_tiles
