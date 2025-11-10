# /// script
# dependencies = ["matplotlib", "numpy", "seaborn"]
# ///

import json
import sys
import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import LogNorm
import seaborn as sns
import pandas as pd

f = open(sys.argv[1], 'r')
d = json.load(f)

# bigtable = numpy.zeros((len(d),len(d)))
# for x, l in enumerate(d):
# 	print()
# 	print(l)
# 	for y, i in enumerate(d[l]):
# 		print(i[1], i[3], i[4])
# 		bigtable[x][y] = i[4]
# fig, ax = plt.subplots()
# # im = ax.imshow(bigtable, norm=LogNorm(vmin=1, vmax=100000))
# im = ax.imshow(bigtable)
# plt.show()


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

count = 0
fig, axes = plt.subplots(ncols=41)
for l in d:
	df = pd.DataFrame(columns=["drug", "score", "match"])
	for i in d[l]:
		# print(i[1], i[3], i[4])
		if i[1] == i[3]:
			match = True
		else:
			match = False
		df = pd.concat([df, pd.DataFrame({"drug" : [i[2]], "score" : [i[4]], "match" : match })])
		# print(i[0], i[2], i[4])
	print(df)
	# sns.stripplot(y=df["score"], hue=df["match"], log_scale=True, ax=axes[count])
	sns.stripplot(y=df["score"], hue=df["match"], ax=axes[count])
	handles, labels = axes[count].get_legend_handles_labels()
	axes[count].legend([],[])

	if count > 39:
		break
	count += 1

plt.tight_layout()
plt.show()


# plt.show()