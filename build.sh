#!/usr/bin/env bash
echo "Building all figures..."
uv run df.py beads --figure beads
uv run df.py aspirin --figure aspirin
uv run df.py variety --figure variety
uv run df.py long --figure long
echo "Done building all figures!"