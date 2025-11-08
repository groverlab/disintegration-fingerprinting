# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///

import json
import sys

f = open(sys.argv[1], 'r')
d = json.load(f)

for l in d:
	print()
	print(l)
	for i in d[l]:
		print(i[1], i[3], i[4])