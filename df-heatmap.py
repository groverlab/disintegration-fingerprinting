import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
from matplotlib.colors import LogNorm
import argparse
import matplotlib.ticker as tkr # Import the ticker module

parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--figure", default="none", help="figure to generate", choices=["aspirin", "variety", "bayer_tylenol_only"],)
args = parser.parse_args()

f = open(args.filename, 'r')
d = json.load(f)


###### Original version with simple labels:
# bayer_tylenol_data = {
# '011': 'Tylenol',
# '012': 'Tylenol',
# '013': 'Tylenol',
# '101': 'Tylenol 2027/03 BC',
# '102': 'Tylenol 2028/04 CA',
# '103': 'Tylenol 2028/04 WA',
# '104': 'Tylenol 2028/04 MD',
# '105': 'Tylenol 2028/04 TN',
# '106': 'Tylenol 2028/04 CO',
# '107': 'Tylenol 2028/05 DC',
# '108': 'Tylenol 2028/07 BC',
# '109': 'Tylenol 2028/10 VA',
# '110': 'Tylenol 2028/12 CO',
# '111': 'Tylenol 2029/01 TN',
# '112': 'Tylenol 2029/01 CO',
# '113': 'Tylenol 2029/01 CA',
# '114': 'Tylenol 2029/02 CA',
# '115': 'Tylenol 2029/03 VA',
# '116': 'Tylenol 2029/03 CA',
# '117': 'Tylenol 2029/07 CO',
# '118': 'Tylenol 2029/08 CA',
# '150': 'Tylenol HOT 1',
# '151': 'Tylenol HOT 2',
# '152': 'Tylenol HOT 3',
# '153': 'Tylenol COLD 1',
# '154': 'Tylenol COLD 2',
# '155': 'Tylenol COLD 3',
# '411': 'Bayer',
# '412': 'Bayer',
# '413': 'Bayer',
# '501': 'Bayer 2026/10 CO',
# '502': 'Bayer 2027/01 CO',
# '503': 'Bayer 2027/06 BC',
# '504': 'Bayer 2027/09 TN',
# '505': 'Bayer 2027/09 CA',
# '506': 'Bayer 2027/10 MD',
# '507': 'Bayer 2027/11 CO',
# '508': 'Bayer 2027/12 BC',
# '509': 'Bayer 2028/01 CA',
# '510': 'Bayer 2028/01 CO',
# '511': 'Bayer 2028/01 TN',
# '512': 'Bayer 2028/01 BC',
# '513': 'Bayer 2028/02 CA',
# '514': 'Bayer 2028/02 DC',
# '515': 'Bayer 2028/02 MN',
# '517': 'Bayer 2028/04 TN',
# '518': 'Bayer 2028/05 VA',
# '519': 'Bayer 2028/06 CA',
# '550': 'Bayer HOT 1',
# '551': 'Bayer HOT 2',
# '552': 'Bayer HOT 3',
# '553': 'Bayer COLD 1',
# '554': 'Bayer COLD 2',
# '555': 'Bayer COLD 3',
# }

##### This version has improved text labels:
bayer_tylenol_data = {  
'101': 'Tylenol 100248 2027/03 British Columbia',
'102': 'Tylenol EFA114 2028/04 California',
'103': 'Tylenol EFA114 2028/04 Washington',
'104': 'Tylenol EFA115 2028/04 Maryland',
'105': 'Tylenol EHA014 2028/04 Tennessee',
'106': 'Tylenol EHA024 2028/04 Colorado',
'107': 'Tylenol DHA053 2028/05 District of Columbia',
'108': 'Tylenol 10032U 2028/07 British Columbia',
'109': 'Tylenol EAA031 2028/10 Virginia',
'110': 'Tylenol GS17273 2028/12 Colorado',
'111': 'Tylenol EBA093 2029/01 Tennessee',
'112': 'Tylenol ECA031 2029/01 Colorado',
'113': 'Tylenol ECA084 2029/01 California',
'114': 'Tylenol EEA032 2029/02 California',
'115': 'Tylenol EEA041 2029/03 Virginia',
'116': 'Tylenol GS17419 2029/03 California',
'117': 'Tylenol EJA107 2029/07 Colorado',
'118': 'Tylenol EMA039 2029/08 California',
'150': 'Tylenol 50 °C for 35 days',
'151': 'Tylenol 50 °C for 35 days',
'152': 'Tylenol 50 °C for 35 days',
'153': 'Tylenol -20 °C for 35 days',
'154': 'Tylenol -20 °C for 35 days',
'155': 'Tylenol -20 °C for 35 days',
'501': 'Bayer NAAE1XL 2026/10 Colorado',
'502': 'Bayer NAAD6DN 2027/01 Colorado',
'503': 'Bayer NAADX4K 2027/06 British Columbia',
'504': 'Bayer NAADWOK 2027/09 Tennessee',
'505': 'Bayer NAADXPL 2027/09 California',
'506': 'Bayer NAAE21H 2027/10 Maryland',
'507': 'Bayer NAAE3NA 2027/11 Colorado',
'508': 'Bayer NAAE4XP 2027/12 British Columbia',
'509': 'Bayer NAAE7F2 2028/01 California',
'510': 'Bayer NAAE870 2028/01 Colorado',
'511': 'Bayer NAAE870 2028/01 Tennessee',
'512': 'Bayer NAAE9EP 2028/01 British Columbia',
'513': 'Bayer NAAE96X 2028/02 California',
'514': 'Bayer NAAEA8L 2028/02 District of Columbia',
'515': 'Bayer NAAEA8L 2028/02 Minnesota',
'517': 'Bayer NAAEECX 2028/04 Tennessee',
'518': 'Bayer NAAELK5 2028/05 Virginia',
'519': 'Bayer NAAEKHE 2028/06 California',
'550': 'Bayer 50 °C for 35 days',
'551': 'Bayer 50 °C for 35 days',
'552': 'Bayer 50 °C for 35 days',
'553': 'Bayer -20 °C for 35 days',
'554': 'Bayer -20 °C for 35 days',
'555': 'Bayer -20 °C for 35 days',
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


if args.figure == "aspirin":
	cbar_ax = ax.inset_axes([0.99, 0.48, 0.04, 0.6])
	heatmapax = sns.heatmap(bigtable, mask=mask, cmap="Blues_r", linewidth=0.5, square=True, cbar_ax=cbar_ax)   # linear scale
	cbar = heatmapax.collections[0].colorbar
	cbar.set_label('← more similar              less similar →', rotation=90, labelpad=-80, size=14)
	cbar.ax.tick_params(labelsize=13)
	plt.text(0.85, 0.95, 'Difference score', horizontalalignment="center", verticalalignment="center", transform=fig.transFigure, size=16)
elif args.figure == "bayer_tylenol_only":
	cbar_ax = ax.inset_axes([1.10, 0.7, 0.04, 0.6])
	heatmapax = sns.heatmap(bigtable, mask=mask, cmap="Blues_r", linewidth=0.5, square=True, cbar_ax=cbar_ax)   # linear scale
	cbar = heatmapax.collections[0].colorbar
	cbar.set_label('← more similar              less similar →', rotation=90, labelpad=-70, size=12)
	cbar.ax.tick_params(labelsize=11)
	plt.text(0.85, 0.97, 'Difference score', horizontalalignment="center", verticalalignment="center", transform=fig.transFigure, size=14)
elif args.figure == "variety":
	formatter = tkr.ScalarFormatter()
	formatter.set_scientific(False)
	cbar_ax = ax.inset_axes([0.94, 0.44, 0.04, 0.6])
	heatmapax = sns.heatmap(bigtable, mask=mask, cmap="Blues_r", linewidth=0.5, norm=LogNorm(), square=True, cbar_ax=cbar_ax, cbar_kws={"format": formatter})   # linear scale
	cbar = heatmapax.collections[0].colorbar
	cbar.set_label('← more similar                           less similar →', rotation=90, labelpad=-75, size=12)
	cbar.ax.tick_params(labelsize=11)
	plt.text(0.85, 0.95, 'Difference score', horizontalalignment="center", verticalalignment="center", transform=fig.transFigure, size=14)


pointer = "— "
for x, l in enumerate(d):
	
	if args.figure == "aspirin":
		label = d[l][1][1].title()
		if label == "Bayer":
			label = "Aspirin (Bayer)"
		elif label == "Generic":
			label = "Aspirin (generic)"
		plt.text(x+1-0.5, x+1-0.5, pointer+label, rotation=45, horizontalalignment="left", verticalalignment="center", rotation_mode='anchor', size=12)
	
	elif args.figure == "bayer_tylenol_only":
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
plt.savefig(args.figure + "_heatmap.png", dpi=600)
plt.savefig(args.figure + "_heatmap.pdf")
