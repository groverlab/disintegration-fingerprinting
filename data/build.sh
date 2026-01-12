#!/usr/bin/env bash
echo "Building all figures..."

uv run ../df-analyze.py beads --figure beads
uv run ../df-analyze.py aspirin --figure aspirin
uv run ../df-analyze.py variety --figure variety
uv run ../df-analyze.py long --figure long

uv run ../df-analyze.py bayer_tylenol/ --figure bayer_tylenol

uv run ../df-analyze.py bayer_tylenol_only/ --figure bayer_tylenol_only
uv run ../df-analyze.py bayer_tylenol_only_subdirs/ --figure bayer_tylenol_only_subdirs

uv run ../df-comparisons.py aspirin\ comparisons\ self.json --figure aspirin
uv run ../df-comparisons.py bayer_tylenol_only\ comparisons\ self.json --figure bayer_tylenol_only
uv run ../df-comparisons.py variety\ comparisons\ self.json --figure variety

uv run ../df-quant.py aspirin\ comparisons.json --figure aspirin
uv run ../df-quant.py bayer_tylenol\ comparisons.json --figure bayer_tylenol
uv run ../df-quant.py variety\ comparisons.json --figure variety

echo "Done building all figures!"