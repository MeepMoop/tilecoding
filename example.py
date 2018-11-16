def example():
  import numpy as np
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  import time

  from tilecoding import TileCoder

  # tile coder dimensions, limits, tilings
  tiles_per_dim = [8, 8]
  lims = [(0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi)]
  tilings = 8

  # create tile coder
  T = TileCoder(tiles_per_dim, lims, tilings)

  # target function with gaussian noise
  def target_ftn(x, y):
    return np.sin(x) + np.cos(y) + 0.1 * np.random.randn()

  # linear function weight vector, step size for SGD
  w = np.zeros(T.n_tiles)
  alpha = 0.1 / tilings

  # take 10,000 samples of target function, output mse of batches of 100 points
  timer = time.time()
  batch_size = 100
  for batches in range(100):
    mse = 0.0
    for b in range(batch_size):
      x = lims[0][0] + np.random.rand() * (lims[0][1] - lims[0][0])
      y = lims[1][0] + np.random.rand() * (lims[1][1] - lims[1][0])
      target = target_ftn(x, y)
      tiles = T[x, y]
      w[tiles] += alpha * (target - w[tiles].sum())
      mse += (target - w[tiles].sum()) ** 2
    mse /= batch_size
    print('samples:', (batches + 1) * batch_size, 'batch_mse:', mse)
  print('elapsed time:', time.time() - timer)

  # get learned function
  print('mapping function...')
  res = 200
  x = np.arange(lims[0][0], lims[0][1], (lims[0][1] - lims[0][0]) / res)
  y = np.arange(lims[1][0], lims[1][1], (lims[1][1] - lims[1][0]) / res)
  z = np.zeros([len(x), len(y)])
  for i in range(len(x)):
    for j in range(len(y)):
      tiles = T[x[i], y[j]]
      z[i, j] = w[tiles].sum()

  # plot
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  X, Y = np.meshgrid(x, y)
  surf = ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'))
  plt.show()

if __name__ == '__main__':
  example()
