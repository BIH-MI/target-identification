#!/bin/bash

# Set the path to your Python script
SCRIPT_PATH="./src/experiment_main.py"

# Run experiments for the Texas dataset with all three outlier detection methods
echo "Running experiments for the Texas dataset with Centroid method..."
python $SCRIPT_PATH --dataset texas --metric centroid

echo "Running experiments for the Texas dataset with Achilles Heel..."
python $SCRIPT_PATH --dataset texas --metric achilles_all_k

echo "Running experiments for the Texas dataset with Log Likelihood method..."
python $SCRIPT_PATH --dataset texas --metric log_likelihood

echo "All experiments completed."
