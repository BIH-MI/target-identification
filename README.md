# target-identification

## Overview
This repository contains the implementation of various outlier detection methods. 

## Installation

### Prerequisites
- Tested for Python 3.11
- Additional dependencies listed in `requirements.txt`.

### Usage

#### Running experiments

To run an experiment with a specific outlier detection method on a chosen dataset, use the experiment_main.py script as follows:
```
python experiment_main.py --dataset <dataset_name> --metric <metric_name> 
```
#### Automated Experiment Script

You can also run all the experiments used in the poster via the provided shell script:
```
./scripts/run_experiments.sh
```