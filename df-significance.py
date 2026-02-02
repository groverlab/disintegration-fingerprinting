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
				uber.append(["Generic aspirin\nvs. Generic aspirin", i[4]])
			elif i[1] == "bayer" and i[3] == "bayer":
				print("BAYER MATCH")
				uber.append(["Bayer aspirin\nvs. Bayer aspirin", i[4]])
			else:
				print("MISMATCH")
				uber.append(["Bayer aspirin\nvs. Generic aspirin", i[4]])
	df = pd.DataFrame(uber, columns=["drugtype", "score"])
	df = df.drop_duplicates()

	order = ["Bayer aspirin\nvs. Bayer aspirin", "Generic aspirin\nvs. Generic aspirin", "Bayer aspirin\nvs. Generic aspirin"]

	fig, axs = plt.subplots(figsize=(7.5, 4))
	plt.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.12)
	
	ax_swarm = sns.stripplot(y="drugtype", x="score", order=order, data=df, log_scale=False, jitter=0.2, size=4)
	sns.boxplot(y="drugtype", x="score", order=order, data=df, log_scale=False, fill=False, showfliers=False)
		
	pairs = [("Bayer aspirin\nvs. Bayer aspirin", "Generic aspirin\nvs. Generic aspirin"), 
			 ("Bayer aspirin\nvs. Bayer aspirin", "Bayer aspirin\nvs. Generic aspirin"),
			 ("Generic aspirin\nvs. Generic aspirin", "Bayer aspirin\nvs. Generic aspirin"),]
	annotator = Annotator(ax_swarm, pairs, data=df, order=order, y="drugtype", x="score", orient='h')
	annotator.configure(test='Mann-Whitney', text_format='star')
	annotator.apply_and_annotate()
	axs.set_xlabel("← more similar                              Difference score                              less similar →")
	axs.set_ylabel("")
	plt.savefig("aspirin_significance.png", dpi=600)


	

	print()
	print(df[["drugtype", "score"]].groupby("drugtype").describe())





######### bayer_tylenol #############
elif args.figure == "bayer_tylenol":
	uber = []
	for x, l in enumerate(d):
		for y, i in enumerate(d[l]):

			# print(x, i[1], y, i[3], i[4])  # short
			print(x, i[0], y, i[2], i[4], end=" ")  # full
			if "TRT" in i[0] and "TRT" in i[2]:
				print("TYLENOL MATCH")
				uber.append(["6. Tylenol vs. Tylenol (different lots)", i[4]])
			elif "ART" in i[0] and "ART" in i[2]:
				print("BAYER MATCH")
				uber.append(["2. Bayer vs. Bayer (different lots)", i[4]])
			elif ("TCOLD" in i[0] and "TRT" in i[2]) or ("TRT" in i[0] and "TCOLD" in i[2]):
				print("COLD TYLENOL MATCH")
				uber.append(["8. Tylenol (-20 °C) vs. Tylenol (room temp.)", i[4]])
			elif ("THOT" in i[0] and "TRT" in i[2]) or ("TRT" in i[0] and "THOT" in i[2]):
				print("HOT TYLENOL MATCH")
				uber.append(["7. Tylenol (50 °C) vs. Tylenol (room temp.)", i[4]])
			elif ("ACOLD" in i[0] and "ART" in i[2]) or ("ART" in i[0] and "ACOLD" in i[2]):
				print("COLD BAYER MATCH")
				uber.append(["4. Bayer (-20 °C) vs. Bayer (room temp.)", i[4]])
			elif ("AHOT" in i[0] and "ART" in i[2]) or ("ART" in i[0] and "AHOT" in i[2]):
				print("HOT BAYER MATCH")
				uber.append(["3. Bayer (50 °C) vs. Bayer (room temp.)", i[4]])
			elif (("ART" in i[0] and "TRT" in i[2]) or ("TRT" in i[0] and "ART" in i[2])):
				print("RT MISMATCH")
				uber.append(["10. Bayer vs. Tylenol (different lots)", i[4]])
			elif ("TCANADA" in i[0] and "TRT" in i[2]) or ("TRT" in i[0] and "TCANADA" in i[2]):
				print("CANADA TYLENOL MATCH")
				uber.append(["9. Tylenol (Canada) vs. Tylenol (US)", i[4]])
			elif ("ACANADA" in i[0] and "ART" in i[2]) or ("ART" in i[0] and "ACANADA" in i[2]):
				print("CANADA BAYER MATCH")
				uber.append(["5. Bayer (Canada) vs. Bayer (US)", i[4]])
			elif "BAYERBOTTLE" in i[0] and "BAYERBOTTLE" in i[2]:
				print("BAYER BOTTLE MATCH")
				uber.append(["1. Bayer vs. Bayer (same lot)", i[4]])
			# elif "GENERICBOTTLE" in i[0] and "GENERICBOTTLE" in i[2]:
			# 	print("GENERIC BOTTLE MATCH")
			# 	uber.append(["generic_bottle", i[4]])
			# elif (("BAYERBOTTLE" in i[0] and "GENERICBOTTLE" in i[2]) or ("GENERICBOTTLE" in i[0] and "BAYERBOTTLE" in i[2])):
			# 	print("BOTTLE MISMATCH")
			# 	uber.append(["bottle_mismatches", i[4]])
			else:
				print("NOTHING YET")

	df = pd.DataFrame(uber, columns=["drugtype", "score"])
	df = df.drop_duplicates()
	# print(df)

	order = ["1. Bayer vs. Bayer (same lot)",
		"2. Bayer vs. Bayer (different lots)",
		"3. Bayer (50 °C) vs. Bayer (room temp.)",
		"4. Bayer (-20 °C) vs. Bayer (room temp.)",
		"5. Bayer (Canada) vs. Bayer (US)",
		"6. Tylenol vs. Tylenol (different lots)",
		"7. Tylenol (50 °C) vs. Tylenol (room temp.)",
		"8. Tylenol (-20 °C) vs. Tylenol (room temp.)",
		"9. Tylenol (Canada) vs. Tylenol (US)",
		"10. Bayer vs. Tylenol (different lots)"]

	fig, axs = plt.subplots(figsize=(8, 4))
	plt.subplots_adjust(left=0.35, right=0.95, top=0.98, bottom=0.13)
	ax_swarm = sns.stripplot(y="drugtype", x="score", data=df, order=order, jitter=0.25, orient='h', size=3)
	ax_box = sns.boxplot(y="drugtype", x="score", data=df, order=order, fill=False, orient='h', showfliers=False)

	pairs = [("1. Bayer vs. Bayer (same lot)", "2. Bayer vs. Bayer (different lots)"),
			 ("2. Bayer vs. Bayer (different lots)", "3. Bayer (50 °C) vs. Bayer (room temp.)"),
			 ("2. Bayer vs. Bayer (different lots)", "4. Bayer (-20 °C) vs. Bayer (room temp.)"),
			 ("2. Bayer vs. Bayer (different lots)", "5. Bayer (Canada) vs. Bayer (US)"),
			 ("6. Tylenol vs. Tylenol (different lots)", "7. Tylenol (50 °C) vs. Tylenol (room temp.)"),
			 ("6. Tylenol vs. Tylenol (different lots)", "8. Tylenol (-20 °C) vs. Tylenol (room temp.)"),
			 ("6. Tylenol vs. Tylenol (different lots)", "9. Tylenol (Canada) vs. Tylenol (US)"), ]
	annotator = Annotator(ax_swarm, pairs, data=df, order=order, y="drugtype", x="score", orient='h')
	annotator.configure(test='Mann-Whitney', text_format='star')
	annotator.apply_and_annotate()
	axs.set_xlabel("← more similar                           Difference score                           less similar →")
	axs.set_ylabel("")
	plt.savefig("bayer_tylenol_significance.png", dpi=600)

	print()
	print(df[["drugtype", "score"]].groupby("drugtype").describe())






######### variety #############
elif args.figure == "variety":
	uber = []
	for x, l in enumerate(d):
		for y, i in enumerate(d[l]):

			print(x, i[1], y, i[3], i[4])  # short
			print(x, i[0], y, i[2], i[4], end=" ")  # full
			if i[1] == i[3]:    # match
				print("MATCH")
				uber.append([i[1], i[4], True])
			else:
				print("MISMATCH")  # mismatch
				uber.append([i[1], i[4], False])

	df = pd.DataFrame(uber, columns=["drugtype", "score", "match"])
	df = df.drop_duplicates()   # included here to avoid repeat points from self-comparisons

	fig, axs = plt.subplots(figsize=(7.5, 9))
	plt.subplots_adjust(left=0.25, right=0.98, top=0.98, bottom=0.05)
	
	ax_swarm = sns.stripplot(y="drugtype", x="score",  data=df[~df["match"]], log_scale=False, jitter=0.3, size=2)
	ax_swarm = sns.stripplot(y="drugtype", x="score",  data=df[df["match"]], log_scale=False, jitter=0.3, size=4)
	# sns.boxplot(y="drugtype", x="score",  data=df, log_scale=False, fill=False, showfliers=False)
	axs.set_xlabel("← more similar                           Difference score                           less similar →")
	axs.set_ylabel("")
	plt.savefig("variety_significance.png", dpi=600)

	print("MISMATCHES:")
	print(df[~df["match"]]["score"].describe())
	print()
	print("MATCHES")
	print(df[df["match"]]["score"].describe())
	print()
	print(df[["drugtype", "score"]].groupby("drugtype").describe())





else:
	print("No figure specified; nothing done!")

