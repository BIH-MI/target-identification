# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pandas as pd
import time
from dataset_configuration import DatasetConfiguration, ATTRIBUTES
from outlier_detection_centroid import OutlierDetectionCentroid
from outlier_detection_log_likelihood import OutlierDetectionLogLikelihood
from outlier_detection_achilles import OutlierDetectionAchillesAllK

# Hardcoded paths and configurations
CONFIG_FILES = {
    'texas': r"data/texas_data_config.yml",
}

DATASET_PATHS = {
    'texas': r'data/texas.csv',
}


def run_experiment(dataset_name, metric):
    """
    Run single experiment with dataset and config corresponding to dataset_name in DATASET_PATHS and CONFIG_FILES.
    """

    config_file = CONFIG_FILES[dataset_name]
    dataset_config = DatasetConfiguration(config_file, target_attributes=ATTRIBUTES.ALL)

    categorical_columns = [x.name for x in dataset_config.get_categorical_attributes()]
    continuous_columns = [x.name for x in dataset_config.get_continuous_attributes()]

    dtype_dict = {col: str for col in categorical_columns}
    dtype_dict.update({col: float for col in continuous_columns})

    path_to_file = DATASET_PATHS[dataset_name]

    dataset = pd.read_csv(path_to_file, sep=";", decimal=".", dtype=dtype_dict, nrows=2000)

    start_time = time.time()

    if metric == 'achilles_all_k':
        detector = OutlierDetectionAchillesAllK(dataset, continuous_columns, categorical_columns)
    elif metric == 'centroid':
        detector = OutlierDetectionCentroid(dataset, continuous_columns, categorical_columns)
    elif metric == 'log_likelihood':
        detector = OutlierDetectionLogLikelihood(dataset, continuous_columns, categorical_columns)
    else:
        raise ValueError("Unsupported metric type.")

    detector.run()

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Save the dataset with outlier metrics added
    results_file_name = f"results/results_{dataset_name}_{metric}.csv"
    detector.dataset.to_csv(results_file_name, sep=";", decimal=".", index=False)
    print(f"Results saved to {results_file_name}.")

    # Prepare log message
    log_message = f"Experiment: {dataset_name}, Metric: {metric}, Time taken: {elapsed_time} seconds\n"
    print(log_message)

    # Define log file name
    log_file_name = f"results/experiment_{dataset_name}_{metric}.log"

    # Save to log file
    with open(log_file_name, 'w') as log_file:
        log_file.write(log_message)

    print(f"Experiment completed in {elapsed_time} seconds. Results saved to {log_file_name}.")
    print(detector.dataset.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run outlier detection experiment.")
    parser.add_argument('--dataset', type=str, choices=['texas', 'adult'], required=True,
                        help="The dataset to use for the experiment.")
    parser.add_argument('--metric', type=str, choices=['centroid', 'log_likelihood', 'achilles_all_k'], required=True,
                        help="The outlier detection metric to use.")
    parser.add_argument('--k', type=int, default=5,
                        help="The number of nearest neighbors to consider (only for achilles).")

    args = parser.parse_args()

    run_experiment(args.dataset, args.metric)
