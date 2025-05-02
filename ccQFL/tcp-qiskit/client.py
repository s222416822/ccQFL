import socket
import time
import pickle
import argparse
import json

from device import Device
from data_preparation import DataPreparation
import os
# NUM_ROUNDS = 100  # Number of communication rounds

devices_list = []
def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--client_id', type=int, required=True, help='Client ID')
    return parser.parse_args()

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def connect_to_server(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    return client_socket

def start_client():
    args = parse_arguments()
    # config = load_config()

    client_id = args.client_id
    # logs_dir = config['logs_dir']

    config = load_config()
    algorithm = config['algorithm']
    data_used = config['data_used']
    NUM_CLIENTS = config['num_devices']
    # NUM_ROUNDS = config['num_rounds']
    maxiter = config['maxiter']

    # logs_dir = f"logs/{algorithm}_{data_used}_maxiter={maxiter}_numDevices={NUM_CLIENTS}"
    # if not os.path.exists(logs_dir):
    #     os.makedirs(logs_dir)

    logs_dir = f"logs/{algorithm}_{data_used}_maxiter={maxiter}_numDevices={NUM_CLIENTS}"
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except FileExistsError:
        pass
    except Exception as e:
        print(f"An error occurred while creating the directory: {e}")

    NUM_ROUNDS = config['num_rounds']
    host = '127.0.0.1'
    port = 8080

    data_preparer = DataPreparation(data_used=config['data_used'])
    devices_data, devices_labels, _, _ = data_preparer.prepare_data()
    device = Device(client_id, devices_data[client_id], devices_labels[client_id], maxiter=config['maxiter'], warm_start=True)
    devices_list.append(device)
    print("[------------------DEVICES LIST----------------]", devices_list)

    socket_client = connect_to_server(host, port)

    # Receive the initial weights from the server
    print(f"Device {client_id} receiving initial weights...")
    initial_weights_data = socket_client.recv(4096)
    initial_weights = pickle.loads(initial_weights_data)
    print(f"Device {client_id} received initial weights: {initial_weights}")

    device.vqc.initial_point = initial_weights

    for round_num in range(NUM_ROUNDS):
        comm_start_time = time.time()

        print(f"Device {client_id} training starting for round {round_num}.....")
        device.training()
        with open(f"{logs_dir}/device.txt", 'a') as file:
            file.write(f"Comm_round: {round_num} - Device: {device.idx}  - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f}\n")

        print(f"Comm_round: {round_num} - Device: {device.idx} - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f}")
        print(f"Device ID: {client_id} completed round {round_num}...")

        weights = device.vqc.weights
        print(f"Device {client_id} sending weights to the server now: {weights}")
        data_to_send = {'weights': weights, 'round': round_num}
        socket_client.sendall(pickle.dumps(data_to_send))
        print(f"Device {client_id} finished sending weights to the server....")

        print(f"Device {client_id} receiving average weights from the server...")
        average_weights_received = socket_client.recv(4096)
        average_weights = pickle.loads(average_weights_received)
        print(f"Device {client_id} received average weights from server: {average_weights}")

        # Update the device with the new average weights
        device.vqc.initial_point = average_weights

        time.sleep(1)  # Wait for 1 second before sending the next message
        print(f"Completed the communication round {round_num}")

        comm_end_time = time.time() - comm_start_time
        with open(f"{logs_dir}/comm_time.txt", 'a') as file:
            file.write(f"Comm_round: {round_num} - Device: {client_id} - Comm_time: {comm_end_time}\n")
        print(f"Communication time for round {round_num} - Device: {client_id} - Comm_time: {comm_end_time} seconds")



    socket_client.close()

    with open(f"{logs_dir}/objective_values_devices.txt", 'a') as file:
        for device in devices_list:
            file.write(f"Device {device.idx}: {device.objective_func_vals}\n")

if __name__ == "__main__":
    start_client()
