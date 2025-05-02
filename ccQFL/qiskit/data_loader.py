from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals
import numpy as np
from genomic_benchmarks.dataset_getters.pytorch_datasets import DemoHumanOrWorm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

algorithm_globals.random_seed = 123

def load_and_preprocess_data(data_used, data_size, subset_size_device, subset_size_server):
    if data_used == "iris":
        num_devices = 3
        iris_data = load_iris()
        features_iris = iris_data.data
        labels_iris = iris_data.target

        plt.rcParams["figure.figsize"] = (6, 6)
        sns.scatterplot(x=features_iris[:, 0], y=features_iris[:, 1], hue=labels_iris, palette="tab10")
        plt.title("IRIS Dataset")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

        alldevices_train_features, server_test_features, alldevices_train_labels, server_test_labels = train_test_split(
            features_iris, labels_iris, train_size=0.9, random_state=algorithm_globals.random_seed
        )

        print(alldevices_train_features.shape)
        print(server_test_features.shape)
        print(alldevices_train_labels.shape)
        print(server_test_labels.shape)
        print(alldevices_train_features)
        print(alldevices_train_labels)
        print(server_test_features)
        print(server_test_labels)

    elif data_used == "fashion":
        num_devices = 10
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        if data_size == "small":
            x_train = x_train[:subset_size_device]
            y_train = y_train[:subset_size_device]
            x_test = x_test[:subset_size_server]
            y_test = y_test[:subset_size_server]
        x_train_flattened = x_train.reshape(x_train.shape[0], -1)
        x_test_flattened = x_test.reshape(x_test.shape[0], -1)
        scaler = StandardScaler()
        x_train_flattened = scaler.fit_transform(x_train_flattened)
        x_test_flattened = scaler.transform(x_test_flattened)
        pca = PCA(n_components=4)
        x_train_pca = pca.fit_transform(x_train_flattened)
        x_test_pca = pca.fit_transform(x_test_flattened)

        plt.rcParams["figure.figsize"] = (8, 8)
        sns.scatterplot(x=x_train_pca[:, 0], y=x_train_pca[:, 1], hue=y_train, palette="tab10", legend="full", s=60, alpha=0.7)
        plt.title("Fashion Dataset")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig("fashion.png")
        plt.savefig("fashion.pdf")
        plt.show()

        plt.rcParams["figure.figsize"] = (8, 8)
        sns.scatterplot(x=x_train_pca[:, 0], y=x_train_pca[:, 1], hue=y_train, palette="tab10", legend="full", s=60, alpha=0.7)
        plt.title("Fashion Dataset")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig("fashion1.png", format='png')
        plt.savefig("fashion1.pdf", format='pdf')
        plt.show()

        print(x_train_flattened.shape)
        print(x_test_flattened.shape)
        print(x_train_pca.shape)
        print(x_test_pca.shape)
        alldevices_train_features, server_test_features, alldevices_train_labels, server_test_labels = train_test_split(
            x_train_pca, y_train, train_size=0.8, random_state=algorithm_globals.random_seed
        )
        print(alldevices_train_features.shape)
        print(server_test_features.shape)
        print(alldevices_train_labels.shape)
        print(server_test_labels.shape)
        print(alldevices_train_features)
        print(alldevices_train_labels)
        print(server_test_features)
        print(server_test_labels)

    elif data_used == "mnist_keras":
        num_devices = 20
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        if data_size == "small":
            x_train = x_train[:subset_size_device]
            y_train = y_train[:subset_size_device]
            x_test = x_test[:subset_size_server]
            y_test = y_test[:subset_size_server]
        x_train_flattened = x_train.reshape(x_train.shape[0], -1)
        x_test_flattened = x_test.reshape(x_test.shape[0], -1)
        scaler = StandardScaler()
        x_train_flattened = scaler.fit_transform(x_train_flattened)
        x_test_flattened = scaler.transform(x_test_flattened)
        pca = PCA(n_components=4)
        x_train_pca = pca.fit_transform(x_train_flattened)
        x_test_pca = pca.fit_transform(x_test_flattened)

        plt.rcParams["figure.figsize"] = (8, 8)
        sns.scatterplot(x=x_train_pca[:, 0], y=x_train_pca[:, 1], hue=y_train, palette="tab10", legend="full", s=60, alpha=0.7)
        plt.title("MNIST Dataset")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.legend(title='Digit', loc='best')
        plt.savefig("mnist.png", dpi=100)
        plt.savefig("mnist.pdf", dpi=100)
        plt.show()

        print(x_train_flattened.shape)
        print(x_test_flattened.shape)
        print(x_train_pca.shape)
        print(x_test_pca.shape)
        alldevices_train_features, server_test_features, alldevices_train_labels, server_test_labels = train_test_split(
            x_train_pca, y_train, train_size=0.8, random_state=algorithm_globals.random_seed
        )
        print(alldevices_train_features.shape)
        print(server_test_features.shape)
        print(alldevices_train_labels.shape)
        print(server_test_labels.shape)
        print(alldevices_train_features)
        print(alldevices_train_labels)
        print(server_test_features)
        print(server_test_labels)

    elif data_used == "mnist":
        num_devices = 10
        mnist_data = load_digits()
        features_mnist = mnist_data.data
        labels_mnist = mnist_data.target
        features_mnist_pca = PCA(n_components=4).fit_transform(features_mnist)

        plt.rcParams["figure.figsize"] = (6, 6)
        sns.scatterplot(x=features_mnist_pca[:, 0], y=features_mnist_pca[:, 1], hue=labels_mnist, palette="tab10")
        plt.title("MNIST Dataset")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()

        alldevices_train_features, server_test_features, alldevices_train_labels, server_test_labels = train_test_split(
            features_mnist, labels_mnist, train_size=0.8, random_state=algorithm_globals.random_seed
        )
        print(alldevices_train_features.shape)
        print(server_test_features.shape)
        print(alldevices_train_labels.shape)
        print(server_test_labels.shape)
        print(alldevices_train_features)
        print(alldevices_train_labels)
        print(server_test_features)
        print(server_test_labels)

    elif data_used == "synthetic":
        num_devices = 10
        num_samples = 1000
        num_features = 4
        features_synthetic = 2 * algorithm_globals.random.random([num_samples, num_features]) - 1
        labels_synthetic = 1 * (np.sum(features_synthetic, axis=1) >= 0)

        plt.rcParams["figure.figsize"] = (6, 6)
        sns.scatterplot(x=features_synthetic[:, 0], y=features_synthetic[:, 1], hue=labels_synthetic, palette="tab10")
        plt.title("Synthetic Dataset")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

        alldevices_train_features, server_test_features, alldevices_train_labels, server_test_labels = train_test_split(
            features_synthetic, labels_synthetic, train_size=0.9, random_state=algorithm_globals.random_seed
        )

    elif data_used == "genomics":
        num_devices = 10
        train_dataset = DemoHumanOrWorm(split='train', version=0)
        train_data_list = list(train_dataset)
        nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        encoded_sequences = []
        labels = []
        for sequence, label in train_data_list:
            encoded_sequence = []
            for nucleotide in sequence:
                encoded_nucleotide = [0] * 8
                if nucleotide in nucleotide_map:
                    index = nucleotide_map[nucleotide]
                    encoded_nucleotide[index] = 1
                encoded_sequence.append(encoded_nucleotide)
            encoded_sequences.append(encoded_sequence)
            labels.append(label)
        features_encoded_sequences_3D_np = np.array(encoded_sequences)
        labels_encoded_sequences_3D_np = np.array(labels)
        print("Encoded Sequences Shape:", features_encoded_sequences_3D_np.shape)
        print("Labels Shape:", labels_encoded_sequences_3D_np.shape)
        encoded_sequences_np_reshaped = features_encoded_sequences_3D_np.reshape(features_encoded_sequences_3D_np.shape[0], -1)
        print("encoded_sequences_np_reshaped", encoded_sequences_np_reshaped.shape)

        plt.rcParams["figure.figsize"] = (6, 6)
        sns.scatterplot(x=encoded_sequences_np_reshaped[:, 0], y=encoded_sequences_np_reshaped[:, 1], hue=labels_encoded_sequences_3D_np, palette="tab10")
        plt.title("Encoded Sequences")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

        alldevices_train_features, server_test_features, alldevices_train_labels, server_test_labels = train_test_split(
            encoded_sequences_np_reshaped, labels_encoded_sequences_3D_np, train_size=0.8, random_state=algorithm_globals.random_seed
        )

    else:
        print("No Data Set Selected!")
        return None, None, None, None, 0

    print(f"Dataset used is: {data_used}")
    return (
        alldevices_train_features,
        server_test_features,
        alldevices_train_labels,
        server_test_labels,
        num_devices
    )