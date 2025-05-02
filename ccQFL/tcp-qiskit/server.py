import socket
import pickle
from threading import Barrier
import threading
import argparse
import json
import numpy as np
from qiskit.circuit.library import RealAmplitudes
import time
import os
from device import Device
from data_preparation import DataPreparation

ansatz = RealAmplitudes(num_qubits=4)
initial_point = np.asarray([0.5] * ansatz.num_parameters)

global aggregated_weights
server_training_event = threading.Event()

def server_thread(server_device, comm_round, aggregated_weights, logs_dir):
    try:
        print("Server performing local training and testing the global model...")
        print("Average initial point for server, ", aggregated_weights)
        server_device.vqc.initial_point = aggregated_weights
        print("Server assigned initial weights..")
        server_device.training()
        print("Server finished training and testing the global model...")
        print(f"Round {comm_round} - Training accuracy: {server_device.train_score_q4:.2f}, Test accuracy: {server_device.test_score_q4:.2f}")
        print("Server finished performing local training and testing....")
        with open(f"{logs_dir}/server.txt", 'a') as file:
            file.write(f"Comm_round: {comm_round} - Device: {server_device.idx}  - train_acc: {server_device.train_score_q4:.2f} - test_acc: {server_device.test_score_q4:.2f}\n")
    except Exception as e:
        print("Server failed to train.")
        print("Error", e)
    finally:
        server_training_event.set()

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    return parser.parse_args()

def handle_client(client_socket, client_address, barrier, round_barrier, num_rounds, num_devices, server_device, all_weights, logs_dir, test_barrier, thread_event):
    global aggregate_weights
    print(f"Connection from {client_address} has been established.")
    print("Server sending initial random weights to all clients....")
    client_socket.send(pickle.dumps(initial_point))

    for comm_round in range(num_rounds):

        barrier.wait()

        try:
            received_data = client_socket.recv(4096)
            print(f"Received local weights from {client_address} in round {comm_round}")
            local_weights_received = pickle.loads(received_data)
            received_weights = local_weights_received['weights']
            print(f"Received weights from client {client_address}: {received_weights}")
            all_weights.append(received_weights)
            print(f"Local weights from all clients so far: {all_weights}")
            print(f"Server: Waiting for all clients to reach round {comm_round}")
            barrier.wait()
            print("Barrier wait finished.")
            print("Server device id:", server_device.idx)

            if len(all_weights) == num_devices:
                print("Server performing aggregation....")
                if len(all_weights) > 0:  # Check to prevent division by zero
                    aggregate_weights = sum(all_weights) / len(all_weights)  # Example aggregation: averaging weights
                    all_weights.clear()  # Clear the list for the next round
                    print(f"Aggregated weights for round {comm_round}: {aggregate_weights}")
                    print("Perform server training and testing...")
                    server_thread(server_device, comm_round, aggregate_weights, logs_dir)
                else:
                    print("Error: No weights to aggregate.")
            else:
                print(f"Error: Expected {num_devices} weights, but got {len(all_weights)}")
            print(f"Server: Wait again to ensure all clients receive the aggregated weights for round {comm_round}")
            barrier.wait()

            print("Server sending aggregated weights to all clients....")
            client_socket.send(pickle.dumps(aggregate_weights))
            print(f"Server: Waiting for all clients to receive the aggregated weights for round {comm_round}")
            round_barrier.wait()
        except Exception as e:
            print(f"Error occurred with client {client_address} in round {comm_round}: {e}")
            break



    print(f"Connection from {client_address} has been closed.")
    client_socket.close()

def main():
    config = load_config()
    algorithm = config['algorithm']
    data_used = config['data_used']
    NUM_CLIENTS = config['num_devices']
    NUM_ROUNDS = config['num_rounds']
    maxiter = config['maxiter']

    logs_dir = f"logs/{algorithm}_{data_used}_maxiter={maxiter}_numDevices={NUM_CLIENTS}"
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except FileExistsError:
        pass
    except Exception as e:
        print(f"An error occurred while creating the directory: {e}")

    host = '127.0.0.1'
    port = 8080
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(NUM_CLIENTS)
    print(f"Server started and listening on port {port}")

    barrier = Barrier(NUM_CLIENTS)
    round_barrier = Barrier(NUM_CLIENTS)
    test_barrier = Barrier(NUM_CLIENTS)
    thread_event = threading.Event()
    all_weights = []

    data_preparer = DataPreparation(data_used=data_used)
    devices_data, devices_labels, server_test_features, server_test_labels = data_preparer.prepare_data()
    server_device = Device(idx=NUM_CLIENTS, data=server_test_features, labels=server_test_labels, maxiter=maxiter, warm_start=True)

    client_threads = []
    for i in range(NUM_CLIENTS):
        client_socket, client_address = server.accept()
        client_handler = threading.Thread(target=handle_client, args=(client_socket, client_address, barrier, round_barrier, NUM_ROUNDS, NUM_CLIENTS, server_device, all_weights, logs_dir, test_barrier, thread_event), name=f"Client-{i}")
        client_handler.start()
        client_threads.append(client_handler)

    for client_handler in client_threads:
        client_handler.join()

    server.close()

    with open(f"{logs_dir}/server_objective_values_devices.txt", 'w') as file:
        file.write(f"Device {server_device.idx}: {server_device.objective_func_vals}\n")

if __name__ == "__main__":
    main()
