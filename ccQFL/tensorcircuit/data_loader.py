import tensorflow as tf
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

def load_and_preprocess_data(dataset_used, datasize_used):
    if dataset_used != "mnist":
        print("Only MNIST dataset is supported!")
        return None, None, None, None

    # Load the MNIST dataset
    (original_x_train, original_y_train), (original_x_test, original_y_test) = tf.keras.datasets.mnist.load_data()

    # Subset the data
    if datasize_used == "40000":
        train_indices = np.random.choice(original_x_train.shape[0], 40000, replace=False)
        test_indices = np.random.choice(original_x_test.shape[0], 2000, replace=False)
        original_x_train = original_x_train[train_indices]
        original_y_train = original_y_train[train_indices]
        original_x_test = original_x_test[test_indices]
        original_y_test = original_y_test[test_indices]

    # Normalize and resize
    encoding_mode = "vanilla"
    no_of_qubits = 10
    no_of_classes = 10

    original_x_train = original_x_train / 255.0
    mean = 0 if encoding_mode == 'vanilla' else (jnp.mean(original_x_train, axis=0) if encoding_mode == 'mean' else 0.5)
    original_x_train = original_x_train - mean
    original_x_train = tf.image.resize(original_x_train[..., tf.newaxis], (int(2 ** (no_of_qubits / 2)), int(2 ** (no_of_qubits / 2)))).numpy()[..., 0].reshape(-1, 2 ** no_of_qubits)
    original_x_train = original_x_train / jnp.sqrt(jnp.sum(original_x_train ** 2, axis=-1, keepdims=True))

    original_x_test = original_x_test / 255.0
    original_x_test = original_x_test - mean
    original_x_test = tf.image.resize(original_x_test[..., tf.newaxis], (int(2 ** (no_of_qubits / 2)), int(2 ** (no_of_qubits / 2)))).numpy()[..., 0].reshape(-1, 2 ** no_of_qubits)
    original_x_test = original_x_test / jnp.sqrt(jnp.sum(original_x_test ** 2, axis=-1, keepdims=True))

    original_y_test = jax.nn.one_hot(original_y_test, no_of_classes)

    return original_x_train, original_y_train, original_x_test, original_y_test

def create_noniid_data(num_devices, n_class, alpha, noniid_type, x_train, y_train):
    x_train, y_train = np.copy(x_train), np.copy(y_train)

    def dirichlet_partitioning(x_data, y_data, num_devices, alpha=0.5):
        class_data = [x_data[y_data == i] for i in np.unique(y_data)]
        class_labels = [y_data[y_data == i] for i in np.unique(y_data)]
        device_data, device_labels = [[] for _ in range(num_devices)], [[] for _ in range(num_devices)]

        for c_data, c_labels in zip(class_data, class_labels):
            proportions = np.random.dirichlet([alpha] * num_devices)
            class_splits = np.split(c_data, (np.cumsum(proportions[:-1]) * len(c_data)).astype(int))
            label_splits = np.split(c_labels, (np.cumsum(proportions[:-1]) * len(c_labels)).astype(int))

            for i, (split_data, split_labels) in enumerate(zip(class_splits, label_splits)):
                device_data[i].append(split_data)
                device_labels[i].append(split_labels)

        device_data = [np.concatenate(d) for d in device_data]
        device_labels = [np.concatenate(l) for l in device_labels]
        return device_data, device_labels

    if noniid_type == "iid":
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, num_devices)
        noniid_data = [(x_train[idx], y_train[idx]) for idx in split_indices]

    elif noniid_type == "single_class":
        noniid_data = [(x_train[y_train == i % 10], y_train[y_train == i % 10]) for i in range(num_devices)]

    elif noniid_type == "partial_class":
        noniid_data = []
        for i in range(num_devices):
            classes = [(i + j) % 10 for j in range(n_class)]
            device_x, device_y = [], []
            for c in classes:
                device_x.append(x_train[y_train == c])
                device_y.append(y_train[y_train == c])
            device_x = np.concatenate(device_x)
            device_y = np.concatenate(device_y)
            noniid_data.append((device_x, device_y))

    elif noniid_type == "class_imbalance":
        noniid_data = []
        for _ in range(num_devices):
            proportions = np.random.dirichlet(np.ones(10), size=1).flatten()
            device_x, device_y = [], []
            for i, p in enumerate(proportions):
                class_data = x_train[y_train == i]
                class_labels = y_train[y_train == i]
                class_size = int(p * len(class_data))
                device_x.append(class_data[:class_size])
                device_y.append(class_labels[:class_size])
            device_x = np.concatenate(device_x)
            device_y = np.concatenate(device_y)
            noniid_data.append((device_x, device_y))

    elif noniid_type == "dirichlet":
        device_data, device_labels = dirichlet_partitioning(x_train, y_train, num_devices, alpha)
        noniid_data = list(zip(device_data, device_labels))

    print(f"\n{noniid_type.capitalize()} Assignment:")
    for i, (device_x, device_y) in enumerate(noniid_data):
        label_distribution = np.bincount(device_y, minlength=10)
        print(f"Device {i+1} - Data size: {len(device_x)}, Label distribution: {label_distribution}")

    plt.figure(figsize=(15, 5))
    for i, (device_x, device_y) in enumerate(noniid_data):
        label_distribution = np.bincount(device_y, minlength=10)
        plt.bar(np.arange(10) + i * 0.1, label_distribution, width=0.1, label=f"Device {i+1}")
    plt.xlabel("Labels")
    plt.ylabel("Frequency")
    plt.title(f"Label Distribution for {noniid_type.capitalize()} Assignment")
    plt.legend()
    plt.show()

    return noniid_data