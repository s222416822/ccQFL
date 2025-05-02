from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals
import numpy as np
from genomic_benchmarks.dataset_getters.pytorch_datasets import DemoHumanOrWorm
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import seaborn as sns


class DataPreparation:
    def __init__(self, data_used="iris", random_seed=123):
        self.data_used = data_used
        self.random_seed = random_seed
        self.num_devices = self._set_num_devices()
        self.maxiter = self._set_maxiter()
        self.features_dataset = None
        self.labels_dataset = None
        self.server_test_features = None
        self.server_test_labels = None
        self.devices_data = []
        self.devices_labels = []

        algorithm_globals.random_seed = self.random_seed

    def _set_num_devices(self):
        if self.data_used == "iris":
            return 3
        elif self.data_used in ["synthetic", "genomics"]:
            return 10
        return 1

    def _set_maxiter(self):
        random_number = 100
        return "100" if random_number == 100 else "random"

    def prepare_data(self):
        if self.data_used == "iris":
            self._prepare_iris_data()
        elif self.data_used == "synthetic":
            self._prepare_synthetic_data()
        elif self.data_used == "genomics":
            self._prepare_genomics_data()
        else:
            print("No Data Set Selected!")

        self._split_data()
        self._distribute_data()

        return self.devices_data, self.devices_labels, self.server_test_features, self.server_test_labels

    def _prepare_iris_data(self):
        iris_data = load_iris()
        self.features_dataset = iris_data.data
        self.labels_dataset = iris_data.target
        self._plot_data(self.features_dataset[:, 0], self.features_dataset[:, 1], self.labels_dataset, "IRIS Dataset")

    def _prepare_synthetic_data(self):
        num_samples = 1000
        num_features = 4
        self.features_dataset = 2 * algorithm_globals.random.random([num_samples, num_features]) - 1
        self.labels_dataset = 1 * (np.sum(self.features_dataset, axis=1) >= 0)
        self._plot_data(self.features_dataset[:, 0], self.features_dataset[:, 1], self.labels_dataset,
                        "Synthetic Dataset")

    def _prepare_genomics_data(self):
        train_dataset = DemoHumanOrWorm(split='train', version=0)
        train_data_list = list(train_dataset)
        nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        encoded_sequences = []
        labels = []
        for sequence, label in train_data_list:
            encoded_sequence = [[0] * 4 for _ in sequence]
            for i, nucleotide in enumerate(sequence):
                if nucleotide in nucleotide_map:
                    encoded_sequence[i][nucleotide_map[nucleotide]] = 1
            encoded_sequences.append(encoded_sequence)
            labels.append(label)

        features_encoded_sequences_3D_np = np.array(encoded_sequences)
        self.labels_dataset = np.array(labels)

        encoded_sequences_np_reshaped = features_encoded_sequences_3D_np.reshape(
            features_encoded_sequences_3D_np.shape[0], -1)
        self.features_dataset = PCA(n_components=4).fit_transform(encoded_sequences_np_reshaped)

        self._plot_data(self.features_dataset[:, 0], self.features_dataset[:, 1], self.labels_dataset,
                        "Encoded Sequences")

    def _plot_data(self, x, y, hue, title):
        plt.rcParams["figure.figsize"] = (6, 6)
        sns.scatterplot(x=x, y=y, hue=hue, palette="tab10")
        plt.title(title)
        # plt.xlabel("Feature 1")
        # plt.ylabel("Feature 2")
        # plt.show()

    def _split_data(self):
        self.alldevices_train_features, self.server_test_features, self.alldevices_train_labels, self.server_test_labels = train_test_split(
            self.features_dataset, self.labels_dataset, train_size=0.8, random_state=self.random_seed)

    def _distribute_data(self):
        combined_data = list(zip(self.alldevices_train_features, self.alldevices_train_labels))
        np.random.shuffle(combined_data)
        shuffled_features, shuffled_labels = zip(*combined_data)
        samples_per_device = len(shuffled_features) // self.num_devices
        remainder = len(shuffled_features) % self.num_devices

        start_index = 0
        for i in range(self.num_devices):
            extra_samples = 1 if i < remainder else 0
            end_index = start_index + samples_per_device + extra_samples
            self.devices_data.append(np.array(shuffled_features[start_index:end_index]))
            self.devices_labels.append(np.array(shuffled_labels[start_index:end_index]))
            start_index = end_index

    def print_device_data(self):
        for i, (data, labels) in enumerate(zip(self.devices_data, self.devices_labels)):
            print(f"Device {i + 1} data:", data)
            print(f"Device {i + 1} labels:", labels)
            print()


# Usage
# data_preparer = DataPreparation(data_used="iris")
# devices_data, devices_labels, server_test_features, server_test_labels = data_preparer.prepare_data()
# data_preparer.print_device_data()
