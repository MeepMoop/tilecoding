# Tile Coding

[Tile coding](https://webdocs.cs.ualberta.ca/~sutton/book/8/node6.html) is a coarse coding function approximation method that uses several overlapping offset grids (tilings) to approximate a continuous space.

# Dependencies

* numpy
* matplotlib (to run the example)

# Usage

```python
from tilecoding import tilecoder

# grid dimensions and tilings
dims = [8, 10, 6, 10]
tilings = 10

# value limits of each dimension (min, max)
lims = [(3.0, 7.5), (-4.4, 4.2), (9.6, 12.7), (0.0, 1.0)]

# create tilecoder with step size 0.1
T = tilecoder(dims, lims, tilings, 0.1)

# training iteration with value 5.5 at location (3.3, -2.1, 11.1, 0.7)
T[3.3, -2.1, 11.1, 0.7] = 5.5

# get approximated value at (3.3, -2.1, 11.1, 0.7)
print T[3.3, -2.1, 11.1, 0.7]
```

# Examples
<p align="center">
  <img src="https://github.com/MeepMoop/tilecoding/blob/master/examples/tc_sincos.png?raw=true"><br>
  8x8 tile coder with 8 tilings approximating f(x, y) = sin(x) + cos(y) + <i>N</i>(0, 0.1)<br><br>
</p>
