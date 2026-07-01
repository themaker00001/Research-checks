#!/usr/bin/env bash
# Reproduce all experiments and figures.
set -e
python3 superposition.py
python3 run_dimensionality_experiment.py
python3 run_feature_geometry_2d.py
python3 run_four_properties.py
echo "All done. See ./results for figures."
