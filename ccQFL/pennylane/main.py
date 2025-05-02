from datetime import datetime
import os
from data_loader import load_and_preprocess_data
from device import Device, multiclass_svm_loss, classify, accuracy
import torch
import numpy as np
import time

# Configuration
num_devices = 3
data_used = "iris_normal"
folder = "pennylane"

def main_method(method, device_data, noniid_type, simulator, X_server_train, X_server_test, y_server_train, y_server_test):
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    logs = f"logs/{simulator}_testing/{date_time}_{method}_devices{num_devices}_noniid={noniid_type}"

    if not os.path.exists(logs):
        os.makedirs(logs)

    device_list = []
    for i, data in enumerate(device_data):
        X = data[:, :-1]
        Y = data[:, -1]
        num_data = Y.shape[0]
        feat_vecs_train, feat_vecs_test, Y_train, Y_test = split_data(X, Y)
        num_train = Y_train.shape[0]
        print("Num params: ", 3 * 6 * int(np.ceil(np.log2(4))) * 3 + 3)  # num_layers=6, feature_size=4
        device = Device(i, feat_vecs_train, Y_train, feat_vecs_test, Y_test, num_train, X, Y, simulator=simulator)
        device_list.append(device)

    average_weights = None
    average_biases = None
    average_params = None
    chained_weights = None
    chained_biases = None
    chained_params = None

    for it in range(100):
        start_time = time.time_ns()
        costs, train_acc, test_acc = [], [], []
        weights_list = []
        biases_list = []

        for i, d in enumerate(device_list):
            if method == "Default":
                if it > 0:
                    d.weights = average_weights
                    d.biases = average_biases
            elif method == "Chained":
                if not it == 0 and i == 0:
                    d.weights = chained_weights
                    d.biases = chained_biases
            batch_index = np.random.randint(0, d.num_train, (10,))  # batch_size=10
            feat_vecs_train_batch = d.feat_vecs_train[batch_index]
            Y_train_batch = d.Y_train[batch_index]
            d.optimizer.zero_grad()
            curr_cost = multiclass_svm_loss(d.q_circuits, d.params, feat_vecs_train_batch, Y_train_batch)
            curr_cost.backward()
            d.optimizer.step()
            predictions_train = classify(d.q_circuits, d.params, d.feat_vecs_train, d.Y_train)
            predictions_test = classify(d.q_circuits, d.params, d.feat_vecs_test, d.Y_test)
            acc_train = accuracy(d.Y_train, predictions_train)
            acc_test = accuracy(d.Y_test, predictions_test)

            print(
                "Comm: {:5d} | Device: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc Test: {:0.7f} "
                "".format(it + 1, i, curr_cost.item(), acc_train, acc_test)
            )
            with open(f"{logs}/local_train.txt", "a") as file:
                file.write(f"Comm: {it+1} - Device {i} - cost_value: {curr_cost.item()} - train_acc: {acc_train}  - test_acc: {acc_test}\n")

            costs.append(curr_cost.item())
            train_acc.append(acc_train)
            test_acc.append(acc_test)

            d.weights, d.biases = d.params
            chained_weights, chained_biases = d.params
            weights_list.append(d.weights)
            biases_list.append(d.biases)

        if method == "Default":
            num_layers1 = len(weights_list[0])
            average_weights = [
                torch.mean(torch.stack([weights[i] for weights in weights_list]), dim=0)
                for i in range(num_layers1)
            ]
            num_biases1 = len(biases_list[0])
            average_biases = [
                torch.mean(torch.stack([biases[i] for biases in biases_list]), dim=0)
                for i in range(num_biases1)
            ]
            average_params = (average_weights, average_biases)
        else:
            average_params = (chained_weights, chained_biases)

        q_circuits = device_list[0].q_circuits
        predictions_test_server = classify(q_circuits, average_params, X_server_test, y_server_test)
        predictions_val_server = classify(q_circuits, average_params, X_server_train, y_server_train)
        acc_test_server = accuracy(y_server_test, predictions_test_server)
        acc_val_server = accuracy(y_server_train, predictions_val_server)
        curr_cost_server = multiclass_svm_loss(q_circuits, average_params, X_server_train, y_server_train)

        print(
            "Server - Comm: {:5d} | test_acc: {:0.7f} | val_acc: {:0.7f} | cost: {:0.7f} "
            "".format(it + 1, acc_test_server, acc_val_server, curr_cost_server.item())
        )

        with open(f"{logs}/server.txt", "a") as file:
            file.write(f"Epoch: {it+1} - acc: {acc_test_server} - cost: {curr_cost_server.item()} - val_acc: {acc_val_server}\n")

        total_time = time.time_ns() - start_time
        with open(f"{logs}/comm_time.txt", "a") as file:
            file.write(f"Comm: {it+1} - Time: {total_time}\n")

def split_data(feature_vecs, Y):
    num_data = len(Y)
    num_train = int(0.75 * num_data)  # train_split=0.75
    index = np.random.permutation(range(num_data))
    feat_vecs_train = feature_vecs[index[:num_train]]
    Y_train = Y[index[:num_train]]
    feat_vecs_test = feature_vecs[index[num_train:]]
    Y_test = Y[index[num_train:]]
    return feat_vecs_train, feat_vecs_test, Y_train, Y_test

if __name__ == "__main__":
    # Load data
    data = load_and_preprocess_data(data_used)
    if data is None:
        print("Failed to load data")
        exit(1)

    # Extract data
    X_server_train = data["X_server_train"]
    X_server_test = data["X_server_test"]
    y_server_train = data["y_server_train"]
    y_server_test = data["y_server_test"]
    iid_data = data["iid_data"]
    single_class_data = data["single_class_data"]
    partial_class_data = data["partial_class_data"]
    imbalanced_data = data["imbalanced_data"]
    dirichlet_data = data["dirichlet_data"]

    # Define methods and configurations
    avg_methods = ["Default", "Chained"]
    noniid_types = ["iid", "single_class", "partial_class", "class_imbalance", "dirichlet"]
    simulators = ["qiskit.remote"]  # Add more if needed: "qiskit.basicsim", "qiskit.aer"

    for noniid_type in noniid_types:
        print("Non-IID Type:", noniid_type)
        if noniid_type == "iid":
            device_data = iid_data
        elif noniid_type == "single_class":
            device_data = single_class_data
        elif noniid_type == "partial_class":
            device_data = partial_class_data
        elif noniid_type == "class_imbalance":
            device_data = imbalanced_data
        elif noniid_type == "dirichlet":
            device_data = dirichlet_data

        from sklearn.preprocessing import MinMaxScaler
        device_data_normalized = device_data

        for avg_method in avg_methods:
            for simulator in simulators:
                print("Simulator:", simulator)
                main_method(avg_method, device_data_normalized, noniid_type, simulator,
                            X_server_train, X_server_test, y_server_train, y_server_test)