# data_loader.py
from sklearn.datasets import load_iris
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
np.random.seed(0)
torch.manual_seed(0)
num_classes = 3
feature_size = 4
train_split = 0.75
num_devices = 3

def load_and_preprocess_data(data_used):
    # Load and normalize data
    # data = np.loadtxt("iris.csv", delimiter=",")
    # X = torch.tensor(data[:, 0:feature_size], dtype=torch.float32)
    # normalization = torch.sqrt(torch.sum(X ** 2, dim=1))
    # X_norm = X / normalization.reshape(len(X), 1)
    # Y = torch.tensor(data[:, -1], dtype=torch.int64)

    # Load the Iris dataset
    if data_used != "iris_normal":
        print("Only iris dataset is supported!")
        return None

    iris = load_iris()
    X = torch.tensor(iris.data, dtype=torch.float32)
    Y = torch.tensor(iris.target, dtype=torch.int64)

    # Normalize the feature matrix
    normalization = torch.sqrt(torch.sum(X ** 2, dim=1))
    X_norm = X / normalization.reshape(-1, 1)

    def split_data(feature_vecs, Y):
        num_data = len(Y)
        num_train = int(train_split * num_data)
        index = np.random.permutation(range(num_data))
        feat_vecs_train = feature_vecs[index[:num_train]]
        Y_train = Y[index[:num_train]]
        feat_vecs_test = feature_vecs[index[num_train:]]
        Y_test = Y[index[num_train:]]
        return feat_vecs_train, feat_vecs_test, Y_train, Y_test

    # Split into train and test
    X_train, X_test, y_train, y_test = split_data(X, Y)
    train_data = torch.cat((X_train, y_train.unsqueeze(1)), dim=1)
    server_test_data = torch.cat((X_test, y_test.unsqueeze(1)), dim=1)

    X_server = server_test_data[:, :-1]
    y_server = server_test_data[:, -1]
    X_server_train, X_server_test, y_server_train, y_server_test = split_data(X_server, y_server)
    server_train_data = torch.cat((X_server_train, y_server_train.unsqueeze(1)), dim=1)
    server_test_data = torch.cat((X_server_test, y_server_test.unsqueeze(1)), dim=1)

    print("Server Train Data:", server_train_data)
    print("Server Test Data:", server_test_data)

    train_data = train_data[torch.randperm(train_data.size(0))]
    iid_data = torch.chunk(train_data, num_devices)

    print("IID Distribution:")
    for i, d in enumerate(iid_data):
        label_distribution = torch.bincount(d[:, -1].to(torch.int64), minlength=len(torch.unique(Y)))
        print(f"Device {i+1} data size: {len(d)}, Label distribution: {label_distribution}")

    # Single-class data setup (Extreme Non-IID)
    single_class_data = [train_data[train_data[:, -1] == i] for i in range(num_devices)]

    # Partial-class data setup (Moderate Non-IID)
    partial_class_data = [
        torch.cat((train_data[train_data[:, -1] == 0], train_data[train_data[:, -1] == 1])),
        torch.cat((train_data[train_data[:, -1] == 1], train_data[train_data[:, -1] == 2])),
        torch.cat((train_data[train_data[:, -1] == 0], train_data[train_data[:, -1] == 2]))
    ]

    # Class-imbalanced data setup (Low Non-IID)
    proportions = [
        [0.7, 0.2, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.3, 0.6]
    ]
    imbalanced_data = []
    for prop in proportions:
        device_data = []
        for i, p in enumerate(prop):
            class_data = train_data[train_data[:, -1] == i]
            class_size = int(p * len(class_data))
            device_data.append(class_data[:class_size])
        imbalanced_data.append(torch.cat(device_data))

    # Dirichlet Distribution-Based Assignment (Controlled Non-IID)

    def dirichlet_partitioning(data, num_devices, alpha=0.5):
        class_indices = [data[data[:, -1] == i] for i in torch.unique(data[:, -1])]
        device_data = [[] for _ in range(num_devices)]
        for class_data in class_indices:
            proportions = np.random.dirichlet([alpha] * num_devices)
            split_sizes = (torch.tensor(proportions) * len(class_data)).int().tolist()
            split_sizes[-1] = len(class_data) - sum(split_sizes[:-1])
            class_splits = torch.split(class_data, split_sizes)
            for i, split in enumerate(class_splits):
                device_data[i].append(split)
        return [torch.cat(parts) for parts in device_data]

    alpha = 0.1
    dirichlet_data = dirichlet_partitioning(train_data, num_devices, alpha)

    print("Single-Class Assignment (Extreme Non-IID):")
    for i, d in enumerate(single_class_data):
        label_distribution = torch.bincount(d[:, -1].to(torch.int64), minlength=num_classes)
        print(f"Device {i+1} data size: {len(d)}, Label distribution: {label_distribution}")

    print("\nPartial-Class Assignment (Moderate Non-IID):")
    for i, d in enumerate(partial_class_data):
        label_distribution = torch.bincount(d[:, -1].to(torch.int64), minlength=num_classes)
        print(f"Device {i+1} data size: {len(d)}, Label distribution: {label_distribution}")

    print("\nClass-Imbalanced Assignment (Low Non-IID):")
    for i, d in enumerate(imbalanced_data):
        label_distribution = torch.bincount(d[:, -1].to(torch.int64), minlength=num_classes)
        print(f"Device {i+1} data size: {len(d)}, Label distribution: {label_distribution}")

    print("\nDirichlet Distribution-Based Assignment (Controlled Non-IID):")
    for i, d in enumerate(dirichlet_data):
        label_distribution = torch.bincount(d[:, -1].to(torch.int64), minlength=num_classes)
        print(f"Device {i+1} data size: {len(d)}, Label distribution: {label_distribution}")

    def plot_label_distribution(data, num_devices, title):
        plt.figure(figsize=(10, 6))
        plt.title(title)
        for i in range(num_devices):
            label_distribution = torch.bincount(data[i][:, -1].to(torch.int64), minlength=num_classes)
            plt.bar(range(num_classes), label_distribution, alpha=0.6, label=f"Device {i+1}")
        plt.xlabel("Labels")
        plt.ylabel("Frequency")
        plt.xticks(range(num_classes), [f"Class {i}" for i in range(num_classes)])
        plt.legend()
        plt.show()

    plot_label_distribution(iid_data, num_devices, "IID Distribution")
    plot_label_distribution(single_class_data, num_devices, "Single-Class Assignment (Extreme Non-IID)")
    plot_label_distribution(partial_class_data, num_devices, "Partial-Class Assignment (Moderate Non-IID)")
    plot_label_distribution(imbalanced_data, num_devices, "Class-Imbalanced Assignment (Low Non-IID)")
    plot_label_distribution(dirichlet_data, num_devices, "Dirichlet Distribution-Based Assignment (Controlled Non-IID)")

    return {
        "X_server_train": X_server_train,
        "X_server_test": X_server_test,
        "y_server_train": y_server_train,
        "y_server_test": y_server_test,
        "iid_data": iid_data,
        "single_class_data": single_class_data,
        "partial_class_data": partial_class_data,
        "imbalanced_data": imbalanced_data,
        "dirichlet_data": dirichlet_data
    }