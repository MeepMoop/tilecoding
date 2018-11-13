# Tile Coding

[Tile coding](http://incompleteideas.net/book/ebook/node88.html#SECTION04232000000000000000) is a coarse coding function approximation method that uses several overlapping offset grids (tilings) to approximate a continuous space.

# Dependencies

* numpy
* matplotlib (to run the example)

# Usage

```python
import numpy as np
from tilecoding import TileCoder

# grid dimensions, value limits of each dimension, and tilings
dims = [8, 10, 6, 10]
lims = [(3.0, 7.5), (-4.4, 4.2), (9.6, 12.7), (0.0, 1.0)]
tilings = 10

# create tilecoder
T = TileCoder(dims, lims, tilings)

# init weights and step size
w = np.zeros(T.n_tiles)
alpha = 0.1 / tilings

# training iteration with value 5.5 at location (3.3, -2.1, 11.1, 0.7)
phi = T[3.3, -2.1, 11.1, 0.7]
w[phi] += alpha * (5.5 - w[phi].sum())

# get approximated value at (3.3, -2.1, 11.1, 0.7)
print(w[phi].sum())
```

# Examples
<p align="center">
  <img src="https://github.com/MeepMoop/tilecoding/blob/master/examples/tc_sincos.png"><br>
  8x8 tile coder with 8 tilings approximating f(x, y) = sin(x) + cos(y) + <i>N</i>(0, 0.1)<br><br>
</p>
