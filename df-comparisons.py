import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
from matplotlib.colors import LogNorm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--figure", default="none", help="figure to generate", choices=["aspirin", "variety", "bayer_tylenol"],)
args = parser.parse_args()

f = open(args.filename, 'r')
d = json.load(f)

bayer_tylenol_data = {
'011': 'Tylenol',
'012': 'Tylenol',
'013': 'Tylenol',
'101': 'Tylenol 2027/03 BC',
'102': 'Tylenol 2028/04 CA',
'103': 'Tylenol 2028/04 WA',
'104': 'Tylenol 2028/04 MD',
'105': 'Tylenol 2028/04 TN',
'106': 'Tylenol 2028/04 CO',
'107': 'Tylenol 2028/05 DC',
'108': 'Tylenol 2028/07 BC',
'109': 'Tylenol 2028/10 VA',
'110': 'Tylenol 2028/12 CO',
'111': 'Tylenol 2029/01 TN',
'112': 'Tylenol 2029/01 CO',
'113': 'Tylenol 2029/01 CA',
'114': 'Tylenol 2029/02 CA',
'115': 'Tylenol 2029/03 VA',
'116': 'Tylenol 2029/03 CA',
'117': 'Tylenol 2029/07 CO',
'150': 'Tylenol HOT 1',
'151': 'Tylenol HOT 2',
'152': 'Tylenol HOT 3',
'153': 'Tylenol COLD 1',
'154': 'Tylenol COLD 2',
'155': 'Tylenol COLD 3',
'411': 'Bayer',
'412': 'Bayer',
'413': 'Bayer',
'501': 'Bayer 2026/10 CO',
'502': 'Bayer 2027/01 CO',
'503': 'Bayer 2027/06 BC',
'504': 'Bayer 2027/09 TN',
'505': 'Bayer 2027/09 CA',
'506': 'Bayer 2027/10 MD',
'507': 'Bayer 2027/11 CO',
'508': 'Bayer 2027/12 BC',
'509': 'Bayer 2028/01 CA',
'510': 'Bayer 2028/01 CO',
'511': 'Bayer 2028/01 TN',
'512': 'Bayer 2028/01 BC',
'513': 'Bayer 2028/02 CA',
'514': 'Bayer 2028/02 DC',
'515': 'Bayer 2028/02 MN',
'516': 'Bayer 2028/04 CA',
'517': 'Bayer 2028/04 TN',
'518': 'Bayer 2028/05 VA',
'519': 'Bayer 2028/06 CA',
'550': 'Bayer HOT 1',
'551': 'Bayer HOT 2',
'552': 'Bayer HOT 3',
'553': 'Bayer COLD 1',
'554': 'Bayer COLD 2',
'555': 'Bayer COLD 3',
}


bigtable = numpy.zeros((len(d),len(d)))
smallest = 1e9
largest = 0
for x, l in enumerate(d):
	print()
	print(l)
	for y, i in enumerate(d[l]):
		# print(x, i[1], y, i[3], i[4])  # short
		print(x, i[0], y, i[2], i[4])  # full
		bigtable[x][y] = i[4]
		if i[4] < smallest:
			smallest = i[4]
		if i[4] > largest:
			largest = i[4]

mask = numpy.triu(numpy.ones_like(bigtable, dtype=bool))
fig, ax = plt.subplots(figsize=(7.5,7.5))
plt.axis("off")
# plt.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.01)
cbar_ax = ax.inset_axes([0.8, 0.5, 0.04, 0.4])
sns.heatmap(bigtable, mask=mask, cmap="Blues_r", linewidth=0.5, norm=LogNorm(), square=True, cbar_ax=cbar_ax)


pointer = "— "
for x, l in enumerate(d):
	if args.figure == "aspirin":
		label = d[l][1][1].title()
		if label == "Bayer":
			label = "Name brand"
		plt.text(x+1-0.5, x+1-0.5, pointer+label, rotation=45, horizontalalignment="left", verticalalignment="center", rotation_mode='anchor', size=12)
	elif args.figure == "bayer_tylenol":
		label = bayer_tylenol_data[d[l][1][0].split(" ")[0][-3:]]
		plt.text(x+1-0.5, x+1-0.5, pointer+label, rotation=45, horizontalalignment="left", verticalalignment="center", rotation_mode='anchor', size=9)
	elif args.figure == "variety":
		label = d[l][1][1].title()
		plt.text(x+1-0.5, x+1-0.5, pointer+label, rotation=45, horizontalalignment="left", verticalalignment="center", rotation_mode='anchor', size=8)
	else:
		plt.text(x+1-0.5, x+1-0.5, pointer+d[l][1][0], rotation=45, horizontalalignment="left", verticalalignment="center", rotation_mode='anchor', size=10)

print("smallest:\t", smallest)
print("largest:\t", largest)

# cax = ax.inset_axes([0.5, 0.1, 0.4, 0.04])
# fig.colorbar(im, cax=cax, orientation="horizontal")
plt.tight_layout()
plt.show()
