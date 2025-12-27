import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy
from matplotlib.colors import LogNorm
import argparse
import bayer_tylenol

parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--figure", default="none", help="figure to generate", choices=["aspirin", "variety", "bayer_tylenol"],)
args = parser.parse_args()

f = open(args.filename, 'r')
d = json.load(f)

##### HEATMAP IMAGE

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
		label = bayer_tylenol.data[d[l][1][0].split(" ")[0][-3:]]
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
exit()






# fig, ax = plt.subplots()
# for x, l in enumerate(d):
# 	print()
# 	print(l)
# 	for y, i in enumerate(d[l]):

# 		print(i[1], i[3], i[4])

# 		if i[1]==i[3]:
# 			plt.plot(x, i[4], "b.", zorder=10)
# 		else:
# 			plt.plot(x, i[4], "r.", zorder=0)

# 		# plt.plot(x, i[4], "o")
# # ax.set_yscale("log")
# plt.show()

# df = pd.DataFrame()
# for l in d:
# 	for i in d[l]:
# 		df = pd.concat([df, pd.DataFrame({i[0] : i[4]}, index=[i[2]])])
# print(df)

# bigtable = numpy.zeros((len(d),len(d)))
# for x, l in enumerate(d):
# 	print()
# 	print(l)
# 	for y, i in enumerate(d[l]):
# 		print(i[1], i[3], i[4])
# 		bigtable[x][y] = i[4]
# ax = sns.swarmplot(bigtable[61])
# # plt.hist(bigtable[9])






############## SWARM PLOTS


# count = 0
# # fig, axes = plt.subplots(ncols=len(d), sharey=True, figsize=(15, 3), constrained_layout=True)
# # fig, axes = plt.subplots(ncols=len(d), sharey=True, figsize=(16, 3))
# fig, axes = plt.subplots(ncols=3, nrows=1, sharey=True, figsize=(9, 9))
# axes = axes.flatten()
# plt.subplots_adjust(left=0.05, right=1, top=1, bottom=0.1, wspace=0)
# for l in d:
# 	print("\n", l)
# 	df = pd.DataFrame(columns=["drug", "score", "match"])
# 	for i in d[l]:
# 		# print(i[1], i[3], i[4])
# 		if i[1] == i[3]:
# 			match = True
# 		else:
# 			match = False
# 		df = pd.concat([df, pd.DataFrame({"drug" : [i[2]], "score" : [i[4]], "match" : match })])
# 		# print(i[0], i[2], i[4])
# 	print(df)
# 	# sns.stripplot(y=df["score"], hue=df["match"], log_scale=True, ax=axes[count])
# 	# sns.stripplot(y=df["score"], hue=df["match"], ax=axes[count])
# 	size = 3.5  ##### good for tylenol and bayer study
# 	size = 2 
# 	size = 5
# 	sns.swarmplot(y=df["score"], hue=df["match"], ax=axes[count], size=size) 

# 	underscore = l.find('_')
# 	axes[count].text(0.5, -0.06, l[underscore+1:underscore+4], horizontalalignment='center', verticalalignment='center', rotation="vertical", transform=axes[count].transAxes)
# 	handles, labels = axes[count].get_legend_handles_labels()
# 	axes[count].get_legend().remove()
# 	if count == 0:
# 		axes[count].spines["top"].set_visible(False)
# 		axes[count].spines["right"].set_visible(False)
# 		axes[count].spines["bottom"].set_visible(False)
# 	else:
# 		axes[count].axis("off")
# 	# if count > 39:
# 	# 	break
# 	count += 1

# # plt.tight_layout()
# plt.show()
