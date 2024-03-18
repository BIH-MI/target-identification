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
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler


class OutlierDetectionLogLikelihood:
    """
    Class to detect outliers based on the Log Likelihood approach.
    A column with rankings will be added to the end of the dataframe.
    """

    def __init__(self, dataset, continuous_columns, categorical_columns):
        scaler = MinMaxScaler()
        self.dataset = dataset
        self.dataset[continuous_columns] = scaler.fit_transform(dataset[continuous_columns])
        self.df_continuous = self.dataset[continuous_columns]
        self.df_categorical = self.dataset[categorical_columns]

        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        # Attributes for pre-calculated values
        self.mean_continuous = None
        self.std_continuous = None
        self.df_categorical_normalized_freq = None

    def preprocess_data(self):
        # Scale continuous columns
        self.mean_continuous = self.df_continuous.mean()
        self.std_continuous = self.df_continuous.std()  # Use ddof=0 for population std

        # Normalize categorical data by frequency

        self.df_categorical_normalized_freq = pd.DataFrame(index=self.df_categorical.index)
        for column in self.df_categorical.columns:
            frequencies = self.df_categorical[column].value_counts(normalize=True)
            self.df_categorical_normalized_freq[column] = self.df_categorical[column].map(frequencies)

    def calculate_log_likelihood_continuous(self, row):
        probabilities = [norm(self.mean_continuous[col], max(self.std_continuous[col], 1e-6)).pdf(row[col]) for col in
                         self.df_continuous.columns]
        probabilities = [max(p, 1e-10) for p in probabilities]  # Avoid log(0)
        return np.sum(np.log(probabilities))

    def calculate_log_likelihood_categorical(self, row):
        probabilities = [max(row[col], 1e-10) for col in self.categorical_columns]  # Ensure no log(0)
        return np.sum(np.log(probabilities))

    def calculate_overall_log_likelihood(self):
        df_continuous_log_likelihood = self.df_continuous.apply(self.calculate_log_likelihood_continuous, axis=1)
        df_categorical_log_likelihood = self.df_categorical_normalized_freq.apply(
            self.calculate_log_likelihood_categorical, axis=1)

        overall_log_likelihood = df_continuous_log_likelihood + df_categorical_log_likelihood
        self.dataset['LogLikelihood'] = overall_log_likelihood
        self.dataset['Probability'] = np.exp(self.dataset['LogLikelihood'])

    def run(self):
        """Run the entire outlier detection process."""
        self.preprocess_data()
        self.calculate_overall_log_likelihood()

# Example usage:
# Assuming 'dataset', 'continuous_columns', and 'categorical_columns' are defined
# detector = OutlierDetectionLogLikelihood(dataset, continuous_columns, categorical_columns)
# detector.run()
# Access the dataset with log likelihoods: detector.dataset
