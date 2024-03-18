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

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class OutlierDetectionAchillesAllK:
    """
    Class to detect outliers based on the Achilles Heel approach (see Achilles’ Heels: Vulnerable Record Identification
    in Synthetic Data Publishing). Columns with rankings will be added to the end of the dataframe for various k-values.
    """

    def __init__(self, dataset, continuous_columns, categorical_columns):
        self.dataset = dataset
        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        self.distances = None

    def preprocess_data(self):
        """Preprocess the dataset by scaling continuous columns and encoding categorical columns."""
        scaler = MinMaxScaler()
        self.dataset[self.continuous_columns] = scaler.fit_transform(self.dataset[self.continuous_columns])
        encoder = OneHotEncoder(sparse_output=False)
        df_categorical_encoded = pd.DataFrame(encoder.fit_transform(self.dataset[self.categorical_columns]))
        df_categorical_encoded.columns = encoder.get_feature_names_out(self.categorical_columns)

        # Combine preprocessed continuous and categorical data
        df_continuous = self.dataset[self.continuous_columns]
        self.dataset_encoded = pd.concat([df_continuous, df_categorical_encoded], axis=1)

    def calculate_distances(self):
        """Calculate custom distances for all records in the dataset using optimized NumPy operations."""
        # Ensure the dataset is a NumPy array
        data_encoded = self.dataset_encoded.to_numpy() if hasattr(self.dataset_encoded,
                                                                  'to_numpy') else self.dataset_encoded
        num_records = data_encoded.shape[0]
        num_original_continuous = len(self.continuous_columns)
        num_original_categorical = len(self.categorical_columns)
        total_attributes = num_original_continuous + num_original_categorical

        # Split continuous and categorical data
        continuous_data = data_encoded[:, :num_original_continuous]
        categorical_data = data_encoded[:, num_original_continuous:]

        # Normalize continuous and categorical data
        norm_continuous_data = np.linalg.norm(continuous_data, axis=1, keepdims=True)
        norm_categorical_data = np.linalg.norm(categorical_data, axis=1, keepdims=True)

        # Avoid division by zero
        norm_continuous_data[norm_continuous_data == 0] = 1
        norm_categorical_data[norm_categorical_data == 0] = 1

        # Calculate cosine similarity matrices
        cosine_sim_continuous = continuous_data @ continuous_data.T / (norm_continuous_data * norm_continuous_data.T)
        cosine_sim_categorical = categorical_data @ categorical_data.T / (
                    norm_categorical_data * norm_categorical_data.T)

        # Normalize similarities
        normalized_continuous = cosine_sim_continuous * (num_original_continuous / total_attributes)
        normalized_categorical = cosine_sim_categorical * (num_original_categorical / total_attributes)

        # Combined distance matrix
        combined_distance = 1 - (normalized_continuous + normalized_categorical)

        # Since distance is symmetric, fill the lower triangle of the distance matrix with values from the upper triangle
        i_lower = np.tril_indices(num_records, -1)
        combined_distance[i_lower] = combined_distance.T[i_lower]

        self.distances = combined_distance

    def detect_outliers(self, k):
        """Detect outliers based on the average distance to k closest neighbors."""
        average_distances_k = np.zeros(self.distances.shape[0])
        for i in range(self.distances.shape[0]):
            sorted_distances = np.sort(self.distances[i])
            average_distances_k[i] = np.mean(sorted_distances[1:k + 1])

        self.dataset[f'AvgDistanceTo{k}Neighbors'] = average_distances_k

    def run(self):
        """Run the entire outlier detection process."""
        self.preprocess_data()
        self.calculate_distances()
        self.detect_outliers(1)
        self.detect_outliers(2)
        self.detect_outliers(3)
        self.detect_outliers(5)
        self.detect_outliers(8)
        self.detect_outliers(13)

# Example usage
# Assuming 'dataset', 'continuous_columns', and 'categorical_columns' are defined
# detector = Achilles(dataset, continuous_columns, categorical_columns, k=5)
# detector.run()
# Access the dataset with distances: detector.dataset
