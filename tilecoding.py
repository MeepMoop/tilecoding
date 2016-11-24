#!/usr/bin/env python
import numpy as np

class tilecoder:
  def __init__(self, dims, limits, tilings, step_size=0.1):
    self._dims = np.array(dims) + 1
    self._limits = np.array(limits)
    self._tilings = tilings
    self._alpha = step_size
    self._n_dims = len(self._dims)
    self._tiling_size = np.prod(self._dims)
    self._hash_vec = np.array([1])
    for i in range(len(dims) - 1):
      self._hash_vec = np.hstack([self._hash_vec, dims[i] * self._hash_vec[-1]])
    self._ranges = self._limits[:, 1] - self._limits[:, 0]
    self._tiles = np.array([0.0] * (self._tilings * self._tiling_size))
    self._offsets = np.arange(float(self._tilings)) / self._tilings

  def _get_tiles(self, x):
    tiles = [0] * self._tilings
    for i in range(self._tilings):
      coords = np.floor(((x - self._limits[:, 0]) / self._ranges) * (self._dims - 1) + self._offsets[i])
      tiles[i] = i * self._tiling_size + int(np.dot(self._hash_vec, coords))
    return tiles

  def _get_val_tiles(self, tiles):
    val = 0
    for i in range(self._tilings):
      val += self._tiles[tiles[i]]
    return val

  def _set_val_tiles(self, tiles, val):
    est = self._get_val_tiles(tiles)
    for i in range(self._tilings):
      self._tiles[tiles[i]] += self._alpha * (val - est) / self._tilings

  def __getitem__(self, x):
    return self._get_val_tiles(self._get_tiles(x))

  def __setitem__(self, x, val):
    self._set_val_tiles(self._get_tiles(x), val)

def example():
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  # tile coder dimensions, limits, and tilings
  dims = [8, 8]
  lims = [(0, 2.0 * np.pi)] * 2
  tilings = 10

  # create swarm
  T = tilecoder(dims, lims, tilings)

  # target function with gaussian noise
  def target_ftn(x, y, noise=True):
    return np.sin(x) + np.cos(y) + noise * np.random.randn() * 0.1

  # randomly sample target function until convergence
  batch_size = 50
  for iters in range(200):
    mse = 0.0
    for b in range(batch_size):
      xi = lims[0][0] + np.random.random() * (lims[0][1] - lims[0][0])
      yi = lims[1][0] + np.random.random() * (lims[1][1] - lims[1][0])
      zi = target_ftn(xi, yi)
      T[xi, yi] = zi
      mse += (T[xi, yi] - zi) ** 2
    mse /= batch_size
    print 'samples:', (iters + 1) * batch_size, 'batch_mse:', mse

  # get learned function
  print 'mapping function...'
  res = 100
  x = np.arange(lims[0][0], lims[0][1], (lims[0][1] - lims[0][0]) / res)
  y = np.arange(lims[1][0], lims[1][1], (lims[1][1] - lims[1][0]) / res)
  z = np.zeros([len(y), len(x)])
  for i in range(len(x)):
    for j in range(len(y)):
      z[j, i] = T[x[i], y[j]]

  # plot
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  X, Y = np.meshgrid(x, y)
  surf = ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'))
  plt.show()

if __name__ == '__main__':
  example()
