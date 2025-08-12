#!/usr/bin/env bash
echo "Building all figures..."
uv run ../df-analyze.py beads --figure beads
uv run ../df-analyze.py aspirin --figure aspirin
uv run ../df-analyze.py variety --figure variety
uv run ../df-analyze.py long --figure long
echo "Done building all figures!"