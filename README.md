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

## Acknowledgments

This project implements methods described in the following papers:

- Meeus M, Guepin F, De Montjoye YA. Achilles’ Heels: Vulnerable Record Identification in Synthetic Data Publishing. In: Computer Security – ESORICS 2023. Cham: Springer Nature Switzerland; 2024. p. 380–99

We acknowledge the authors of these papers for their contributions and recommend referencing their work according to academic standards if you use their metric in your research.

## Funding

This work was, in parts, funded by the German Minstry of Health as part of the KI-FDZ project (grant agreement number 2521DAT01C).


## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
