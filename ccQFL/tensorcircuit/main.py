from datetime import datetime
import os
from data_loader import load_and_preprocess_data, create_noniid_data
from device import Device, compute_loss, compute_accuracy, pred
import jax.numpy as jnp
import tensorflow as tf
from tqdm import tqdm
import time
from jax import random
import optax
import jax

# Configuration
num_devices = 20  # n_node
no_of_classes = 10
no_of_qubits = 10
k = 42
comm_rounds = 10
n_class = 3
dataset_used = "mnist"
datasize_used = "40000"
randClass = f"nClass={n_class}"
methods = ["Default", "Chained"]
noniid_types = ["iid", "partial_class", "class_imbalance", "dirichlet"]
alpha = 0.5

def main_method(method, noniid_data, noniid_type, original_x_test, original_y_test):
    overall_start_time = time.time()
    devices_list = generate_devices(noniid_data)

    experiment = f"{method}_{datasize_used}_{dataset_used}_{no_of_qubits}q_{num_devices}n_{k}k_{randClass}_r={comm_rounds}noniid={noniid_type}"
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    logs = f"logs/{date_time}_{experiment}"

    if not os.path.exists(logs):
        os.makedirs(logs)

    if method == "Default":
        avg_params = None
        params_list = []
        print("Start Communication Rounds")
        for b in range(comm_rounds):
            current_time = time.time_ns()
            print(f"------------------COMM ROUND: {b} - Start training Device")
            print("Device Training Method.... ")
            for device in devices_list:
                local_epochs = 2
                print(f"=====================Device {device.id} training start...")
                print(f"Device {device.id} worker training start...\n")
                print(f"*************************Logs folder value:, {logs}")
                data_train_length = len(device.data_train)
                results = []

                if b > 0:
                    device.params = avg_params

                for epoch in tqdm(range(local_epochs), leave=False):
                    print(f"Epoch : {epoch}")
                    highest_acc = 0
                    total_acc = 0
                    num_records = 0
                    average_acc = 0
                    last_acc = 0

                    for i, (x, y) in enumerate(device.data_train):
                        x, y = x.numpy(), y.numpy()
                        loss_val, grad_val = compute_loss(device.params, x, y, k)
                        updates, device.opt_state = device.opt.update(grad_val, device.opt_state, device.params)
                        device.params = optax.apply_updates(device.params, updates)

                        if i % 5 == 0:
                            loss_mean = jnp.mean(loss_val)
                            acc = jnp.mean(compute_accuracy(device.params, x, y, k))
                            tqdm.write(f'world {b}, epoch {epoch}, {i}/{data_train_length}: loss={loss_mean:.4f}, acc={acc:.4f}')
                            total_acc += acc
                            num_records += 1
                            last_acc = acc
                            if acc > highest_acc:
                                highest_acc = acc
                            print(f"Iteration {i} - Total accuracy: {total_acc}; Num records: {num_records}")

                    print(f"Highest accuracy: {highest_acc}")
                    average_acc = total_acc / num_records
                    print(f"Device {device.id} training Epoch: {epoch} done...")
                    results.append(f"Comm: {b} - Device {device.id} - train_loss: {loss_mean} - highest_train_acc: {highest_acc:.2f} - avg_train_acc: {average_acc:.2f} - last_acc: {last_acc}\n")
                    print(f"Device {device.id} training Epoch: {epoch} done, Highest Train Acc: {highest_acc:.2f}, Average Train Acc: {average_acc:.2f}")

                try:
                    with open(f"{logs}/train_results.txt", "a") as file:
                        print(f"Saving results to {logs}/train_results.txt...")
                        file.writelines(results)
                        print("Results successfully saved.")
                except Exception as e:
                    print(f"Failed to save results to file: {e}")

                params_list.append(device.params)
                print(f"Device {device.id} work COMPLETE")

            print(f"------------------COMM ROUND: {b} - Start Server Task")
            print("Server Task Start QFL")
            avg_params = jnp.mean(jnp.stack(params_list, axis=0), axis=0)
            test_acc = jnp.mean(pred(avg_params, original_x_test[:100], k).argmax(axis=-1) == original_y_test[:100].argmax(axis=-1))
            test_loss = -jnp.mean(jnp.log(pred(avg_params, original_x_test[:100], k)) * original_y_test[:100])
            tqdm.write(f'Comm {b}: Server Test Acc={test_acc:.4f}, Server Test Loss={test_loss:.4f}')
            with open(f"{logs}/server_test_results.txt", "a") as file:
                file.write(f"Comm: {b} - test_loss: {test_loss} - test_acc: {test_acc}\n")

            print(f"------------------COMM ROUND: {b} - Finish Server Task")
            final_time = time.time_ns() - current_time
            with open(f"{logs}/global_comm_time.txt", "a") as file:
                file.write(f"METHOD: {method} - COMM ROUND: {b} - Finish Comm Round - Comm TIME: {final_time}\n")
            print(f"METHOD: {method} - COMM ROUND: {b} - Finish Comm Round - Comm TIME: {final_time}")

            with open(f"{logs}/comm_time.txt", "a") as file:
                file.write(f"Comm: {b} - Time: {final_time}\n")

    else:  # Chained
        chained_params = None
        for b in range(comm_rounds):
            current_time = time.time_ns()
            print(f"Communication Round: {b}")
            print(f"COMM ROUND: {b} - Start training Device")
            print("Device Training Method.... ")

            for i, device in enumerate(devices_list):
                local_epochs = 2
                print(f"=====================Device {device.id} training start...")
                if not b == 0 and i == 0:
                    device.params = chained_params

                print(f"Device {device.id} worker training start...\n")
                print(f"*************************Logs folder value:, {logs}")
                data_train_length = len(device.data_train)
                results = []

                for epoch in tqdm(range(local_epochs), leave=False):
                    print(f"Epoch : {epoch}")
                    highest_acc = 0
                    total_acc = 0
                    num_records = 0
                    average_acc = 0
                    last_acc = 0

                    for i, (x, y) in enumerate(device.data_train):
                        x, y = x.numpy(), y.numpy()
                        loss_val, grad_val = compute_loss(device.params, x, y, k)
                        updates, device.opt_state = device.opt.update(grad_val, device.opt_state, device.params)
                        device.params = optax.apply_updates(device.params, updates)

                        if i % 5 == 0:
                            loss_mean = jnp.mean(loss_val)
                            acc = jnp.mean(compute_accuracy(device.params, x, y, k))
                            tqdm.write(f'world {b}, epoch {epoch}, {i}/{data_train_length}: loss={loss_mean:.4f}, acc={acc:.4f}')
                            total_acc += acc
                            num_records += 1
                            last_acc = acc
                            if acc > highest_acc:
                                highest_acc = acc
                            print(f"Iteration {i} - Total accuracy: {total_acc}; Num records: {num_records}")

                    print(f"Highest accuracy: {highest_acc}")
                    average_acc = total_acc / num_records
                    print(f"Device {device.id} training Epoch: {epoch} done...")
                    results.append(f"Comm: {b} - Device {device.id} - train_loss: {loss_mean} - highest_train_acc: {highest_acc:.2f} - avg_train_acc: {average_acc:.2f} - last_acc: {last_acc}\n")
                    print(f"Device {device.id} training Epoch: {epoch} done, Highest Train Acc: {highest_acc:.2f}, Average Train Acc: {average_acc:.2f}")

                try:
                    with open(f"{logs}/train_results.txt", "a") as file:
                        print(f"Saving results to {logs}/train_results.txt...")
                        file.writelines(results)
                        print("Results successfully saved.")
                except Exception as e:
                    print(f"Failed to save results to file: {e}")

                print(f"Device {device.id} work COMPLETE")
                chained_params = device.params

            print(f"COMM ROUND: {b} - Start Server Task")
            final_time = time.time_ns() - current_time
            with open(f"{logs}/global_comm_time.txt", "a") as file:
                file.write(f"METHOD: {method} - COMM ROUND: {b} - Finish Comm Round - Comm TIME: {final_time}\n")
            print(f"METHOD: {method} - COMM ROUND: {b} - Finish Comm Round - Comm TIME: {final_time}")

            with open(f"{logs}/comm_time.txt", "a") as file:
                file.write(f"Comm: {b} - Time: {final_time}\n")

            print("Server Task Start PQFL")
            avg_params = chained_params
            test_acc = jnp.mean(pred(avg_params, original_x_test[:100], k).argmax(axis=-1) == original_y_test[:100].argmax(axis=-1))
            test_loss = -jnp.mean(jnp.log(pred(avg_params, original_x_test[:100], k)) * original_y_test[:100])
            tqdm.write(f'Comm. {b}: Server Test Acc={test_acc:.4f}, Server Test Loss={test_loss:.4f}')
            with open(f"{logs}/server_test_results.txt", "a") as file:
                file.write(f"Comm: {b} - test_loss: {test_loss} - test_acc: {test_acc}\n")

def generate_devices(device_data):

    key = random.PRNGKey(42)
    device_set = {}
    opt = optax.adam(learning_rate=1e-2)

    for node, (device_x, device_y) in enumerate(device_data):
        deviceId = node
        device_y = jax.nn.one_hot(device_y, no_of_classes)
        data_train = tf.data.Dataset.from_tensor_slices((device_x, device_y)).batch(128)
        key, subkey = random.split(key)
        params = random.normal(subkey, (3 * k, no_of_qubits))
        opt_state = opt.init(params)
        device_set[node] = Device(deviceId, data_train, params, opt_state)
        print(f"Device {node} - Classes: {np.unique(device_y.argmax(axis=-1))}")

    return list(device_set.values())

if __name__ == "__main__":
    # Load and preprocess data
    original_x_train, original_y_train, original_x_test, original_y_test = load_and_preprocess_data(dataset_used, datasize_used)

    for noniid_type in noniid_types:
        print(f"Non-IID Type: {noniid_type}")
        noniid_data = create_noniid_data(num_devices, n_class, alpha, noniid_type, original_x_train, original_y_train)
        for method in methods:
            print(f"Running method: {method}")
            main_method(method, noniid_data, noniid_type, original_x_test, original_y_test)