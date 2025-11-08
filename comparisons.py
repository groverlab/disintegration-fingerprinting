# /// script
# requires-python = ">=3.14"
# dependencies = ["matplotlib", "numpy"]
# ///

import json
import sys
import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import LogNorm

f = open(sys.argv[1], 'r')
d = json.load(f)
bigtable = numpy.zeros((len(d),len(d)))

for x, l in enumerate(d):
	print()
	print(l)
	for y, i in enumerate(d[l]):
		print(i[1], i[3], i[4])
		bigtable[x][y] = i[4]


fig, ax = plt.subplots()
im = ax.imshow(bigtable, norm=LogNorm(vmin=1, vmax=100000))
plt.show()