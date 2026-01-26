import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import use
import argparse
import scipy.stats
from statannotations.Annotator import Annotator

use("MacOSX")
plt.rcParams["font.family"] = "Arial"
sns.set_style("whitegrid")

parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--figure", default="none", help="figure to generate", choices=["aspirin", "variety", "bayer_tylenol"],)
args = parser.parse_args()

if "self" in args.filename:
	sys.exit("ERROR:  run this on non-self comparisons only")

f = open(args.filename, 'r')
d = json.load(f)









######### aspirin #############
if args.figure == "aspirin":
	uber = []
	for x, l in enumerate(d):
		for y, i in enumerate(d[l]):
			print(x, i[1], y, i[3], i[4], end=" ")  # short
			# print(x, i[0], y, i[2], i[4], end=" ")  # full
			if i[1] == "generic" and i[3] == "generic":
				print("GENERIC MATCH")
				uber.append(["Pairs of\ngeneric aspirins", i[4]])
			elif i[1] == "bayer" and i[3] == "bayer":
				print("BAYER MATCH")
				uber.append(["bayer_match", i[4]])
			else:
				print("MISMATCH")
				uber.append(["mismatch", i[4]])
	df = pd.DataFrame(uber, columns=["drugtype", "score"])
	df = df.drop_duplicates()

	order = ["bayer_match", "Pairs of\ngeneric aspirins", "mismatch"]

	fig, axs = plt.subplots(figsize=(7.5, 5))
	plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.12)
	
	ax_swarm = sns.stripplot(y="drugtype", x="score", order=order, data=df, log_scale=False, jitter=0.2, size=4)
	sns.boxplot(y="drugtype", x="score", order=order, data=df, log_scale=False, fill=False, showfliers=False)
		
	pairs = [("bayer_match", "Pairs of\ngeneric aspirins"), ]
	annotator = Annotator(ax_swarm, pairs, data=df, order=order, y="drugtype", x="score", orient='h')
	annotator.configure(test='Mann-Whitney', text_format='star')
	annotator.apply_and_annotate()
	axs.set_xlabel("← more similar                              Difference score                              less similar →")
	axs.set_ylabel("")
	plt.savefig("aspirin_significance.png", dpi=600)







######### bayer_tylenol #############
elif args.figure == "bayer_tylenol":
	uber = []
	for x, l in enumerate(d):
		for y, i in enumerate(d[l]):

			# print(x, i[1], y, i[3], i[4])  # short
			print(x, i[0], y, i[2], i[4], end=" ")  # full
			if "TRT" in i[0] and "TRT" in i[2]:
				print("TYLENOL MATCH")
				uber.append(["tylenols", i[4]])
			elif "ART" in i[0] and "ART" in i[2]:
				print("BAYER MATCH")
				uber.append(["bayers", i[4]])
			elif ("TCOLD" in i[0] and "TRT" in i[2]) or ("TRT" in i[0] and "TCOLD" in i[2]):
				print("COLD TYLENOL MATCH")
				uber.append(["cold_tylenols", i[4]])
			elif ("THOT" in i[0] and "TRT" in i[2]) or ("TRT" in i[0] and "THOT" in i[2]):
				print("HOT TYLENOL MATCH")
				uber.append(["hot_tylenols", i[4]])
			elif ("ACOLD" in i[0] and "ART" in i[2]) or ("ART" in i[0] and "ACOLD" in i[2]):
				print("COLD BAYER MATCH")
				uber.append(["cold_bayers", i[4]])
			elif ("AHOT" in i[0] and "ART" in i[2]) or ("ART" in i[0] and "AHOT" in i[2]):
				print("HOT BAYER MATCH")
				uber.append(["hot_bayers", i[4]])
			elif (("ART" in i[0] and "TRT" in i[2]) or ("TRT" in i[0] and "ART" in i[2])):
				print("RT MISMATCH")
				uber.append(["mismatches", i[4]])
			elif ("TCANADA" in i[0] and "TRT" in i[2]) or ("TRT" in i[0] and "TCANADA" in i[2]):
				print("CANADA TYLENOL MATCH")
				uber.append(["canadian_tylenols", i[4]])
			elif ("ACANADA" in i[0] and "ART" in i[2]) or ("ART" in i[0] and "ACANADA" in i[2]):
				print("CANADA BAYER MATCH")
				uber.append(["canadian_bayers", i[4]])
			elif "BAYERBOTTLE" in i[0] and "BAYERBOTTLE" in i[2]:
				print("BAYER BOTTLE MATCH")
				uber.append(["bayer_bottle", i[4]])
			elif "GENERICBOTTLE" in i[0] and "GENERICBOTTLE" in i[2]:
				print("GENERIC BOTTLE MATCH")
				uber.append(["generic_bottle", i[4]])
			elif (("BAYERBOTTLE" in i[0] and "GENERICBOTTLE" in i[2]) or ("GENERICBOTTLE" in i[0] and "BAYERBOTTLE" in i[2])):
				print("BOTTLE MISMATCH")
				uber.append(["bottle_mismatches", i[4]])
			else:
				print("NOTHING YET")

	df = pd.DataFrame(uber, columns=["type", "score"])
	df = df.drop_duplicates()
	# print(df)

	order = ["bayer_bottle", "generic_bottle", "bottle_mismatches",
		"bayers", "hot_bayers", "cold_bayers", "canadian_bayers",
		"tylenols", "hot_tylenols", "cold_tylenols", "canadian_tylenols", "mismatches"]

	# bayer_bottle = list(df[df.type == "bayer_bottle"]["score"])
	# generic_bottle =list(df[df.type == "generic_bottle"]["score"])
	# bottle_mismatches = list(df[df.type == "bottle_mismatches"]["score"])
	# bayers = list(df[df.type == "bayers"]["score"])
	# hot_bayers = list(df[df.type == "hot_bayers"]["score"])
	# cold_bayers = list(df[df.type == "cold_bayers"]["score"])
	# canadian_bayers = list(df[df.type == "canadian_bayers"]["score"])
	# tylenols = list(df[df.type == "tylenols"]["score"])
	# hot_tylenols = list(df[df.type == "hot_tylenols"]["score"])
	# cold_tylenols = list(df[df.type == "cold_tylenols"]["score"])
	# canadian_tylenols = list(df[df.type == "canadian_tylenols"]["score"])
	# mismatches = list(df[df.type == "mismatches"]["score"])

	# # result = scipy.stats.kstest(cold_bayers, bottle_mismatches)
	# result = scipy.stats.mannwhitneyu(tylenols, canadian_tylenols)
	# print(result)

	# print(bayer_bottle)
	# print(generic_bottle)

	fig, axs = plt.subplots(figsize=(8, 4))
	plt.subplots_adjust(left=0.20, right=0.95, top=0.98, bottom=0.13)
	ax_swarm = sns.stripplot(y="type", x="score", data=df, order=order, jitter=0.25, orient='h', size=3)
	# ax_box = sns.boxplot(y="type", x="score", data=df, order=order, fill=False, orient='h', showfliers=False)

	pairs = [("bayer_bottle", "generic_bottle"), ("bayer_bottle", "bayers"),
			 ("bayers", "hot_bayers"), ("bayers", "cold_bayers"), ("bayers", "canadian_bayers"),
			 ("tylenols", "hot_tylenols"), ("tylenols", "cold_tylenols"), ("tylenols", "canadian_tylenols"), ]
	annotator = Annotator(ax_swarm, pairs, data=df, order=order, y="type", x="score", orient='h')
	annotator.configure(test='Mann-Whitney', text_format='star')
	annotator.apply_and_annotate()
	axs.set_xlabel("← more similar                           Difference score                           less similar →")
	axs.set_ylabel("")
	plt.savefig("bayer_tylenol_significance.png", dpi=600)





######### variety #############
elif args.figure == "variety":
	uber = []
	for x, l in enumerate(d):
		for y, i in enumerate(d[l]):

			print(x, i[1], y, i[3], i[4])  # short
			# print(x, i[0], y, i[2], i[4], end=" ")  # full
			if i[1] == i[3]:    # match
				print("MATCH")
				uber.append([i[1], i[4], True])
			else:
				print("MISMATCH")  # mismatch
				uber.append([i[1], i[4], False])

	df = pd.DataFrame(uber, columns=["drugtype", "score", "match"])
	df = df.drop_duplicates()

	fig, axs = plt.subplots(figsize=(7.5, 9))
	plt.subplots_adjust(left=0.25, right=0.98, top=0.98, bottom=0.05)
	
	ax_swarm = sns.stripplot(y="drugtype", x="score",  data=df[~df["match"]], log_scale=False, jitter=0.3, size=2)
	ax_swarm = sns.stripplot(y="drugtype", x="score",  data=df[df["match"]], log_scale=False, jitter=0.3, size=4)
	# sns.boxplot(y="drugtype", x="score",  data=df, log_scale=False, fill=False, showfliers=False)
	axs.set_xlabel("← more similar                           Difference score                           less similar →")
	axs.set_ylabel("")
	plt.savefig("variety_significance.png", dpi=600)

	print()
	print("MISMATCHES:")
	print("mean", df[~df["match"]]["score"].mean())
	print("median", df[~df["match"]]["score"].median())
	print("std", df[~df["match"]]["score"].std())
	print()
	print("MATCHES")
	print("mean", df[df["match"]]["score"].mean())
	print("median", df[df["match"]]["score"].median())
	print("std", df[df["match"]]["score"].std())





else:
	print("No figure specified; nothing done!")

