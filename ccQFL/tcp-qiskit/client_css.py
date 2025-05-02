import socket
import pickle
import threading
import argparse
import json
import time
import numpy as np
from qiskit.circuit.library import RealAmplitudes
import os
from datetime import datetime
from device import Device
from data_preparation import DataPreparation

ansatz = RealAmplitudes(num_qubits=4)

# Synchronization events for managing the training sequence
train_events = []
stop_event = threading.Event()
server_training_event = threading.Event()
server_done_event = threading.Event()  # New event for server completion

peer_weights = {}
num_devices = 0
devices_list = []

def server_thread(server_device, comm_round, aggregated_weights, logs_dir):
    try:
        print("Server performing local training and testing the global model...")
        print("Average initial point for server, ", aggregated_weights)
        server_device.vqc.initial_point = aggregated_weights
        print("Server assigned initial weights..")
        print("Trying...")
        print(type(server_device))
        print(server_device.vqc.initial_point)
        print("ID-----------", server_device.idx)
        server_device.training()
        print("Server finished training and testing the global model...")
        print(
            f"Round {comm_round} - Training accuracy: {server_device.train_score_q4:.2f}, Test accuracy: {server_device.test_score_q4:.2f}")
        print("Server finished performing local training and testing....")
        print("Writing to server text file... ")

        with open(f"{logs_dir}/server.txt", 'a') as file:
            file.write(
                f"Comm_round: {comm_round} - Device: {server_device.idx}  - train_acc: {server_device.train_score_q4:.2f} - test_acc: {server_device.test_score_q4:.2f}\n")

    except Exception as e:
        print("Server failed to train.")
        print("Error", e)

    finally:
        # Signal that server training has finished
        server_training_event.set()
        server_done_event.set()  # Signal that the server is done


def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--client_id', type=int, required=True, help='Client ID')
    return parser.parse_args()


def load_config():
    with open('config_css.json', 'r') as f:
        return json.load(f)


def start_server(client_id, port, threads_list):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('127.0.0.1', port))
    server_socket.listen()
    print(f"[START SERVER] Client {client_id} listening on port {port}...")

    while not stop_event.is_set():
        server_socket.settimeout(1)
        try:
            client_socket, client_address = server_socket.accept()
            print(f"[START SERVER] Client {client_id} accepted connection from {client_address}")
            thread = threading.Thread(target=handle_peer, args=(client_socket,))
            thread.start()
            threads_list.append(thread)
        except socket.timeout:
            continue
    server_socket.close()


def handle_peer(client_socket):
    try:
        data = client_socket.recv(4096)
        if data:
            received_data = pickle.loads(data)
            if received_data['type'] == 'weights':
                client_id = received_data['client_id']
                round_num = received_data['round']
                weights = received_data['weights']
                peer_weights[client_id] = (round_num, weights)
                print(f"[HANDLE PEER] Received weights from peer {client_id} for round {round_num}")
                print(f"Received weights: {weights}")
                next_client_id = (client_id + 1) % num_devices
                train_events[next_client_id].set()
    except Exception as e:
        print(f"Error receiving data: {e}")
    finally:
        client_socket.close()


def connect_to_peer(peer):
    while not stop_event.is_set():
        try:
            peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            peer_socket.connect((peer['host'], peer['port']))
            print(f"[CONNECT TO PEER] Connected to peer at {peer['host']}:{peer['port']}")
            return peer_socket
        except ConnectionRefusedError:
            print(f"[CONNECT TO PEER] Connection to {peer['host']}:{peer['port']} refused, retrying in 1 second...")
            time.sleep(1)
    return None


def send_to_peer(peer, data):
    while not stop_event.is_set():
        try:
            peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            peer_socket.connect((peer['host'], peer['port']))
            peer_socket.sendall(pickle.dumps(data))
            print(f"[SEND TO PEER] Sent data to peer at {peer['host']}:{peer['port']}: {data}")
            peer_socket.close()
            return
        except Exception as e:
            print(f"Error sending data: {e}")
            time.sleep(1)



def main():
    global num_devices, peer_weights

    args = parse_arguments()
    client_id = args.client_id

    config = load_config()
    algorithm = config['algorithm']
    data_used = config['data_used']
    num_devices = config['num_devices']
    NUM_ROUNDS = config['num_rounds']
    peers = config['peers']
    maxiter = config['maxiter']

    logs_dir = f"logs/{algorithm}_{data_used}_maxiter={maxiter}_numDevices={num_devices}"
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except FileExistsError:
        pass
    except Exception as e:
        print(f"An error occurred while creating the directory: {e}")

    for _ in range(num_devices):
        train_events.append(threading.Event())

    # setup_logging(logs_dir)
    data_preparer = DataPreparation(data_used=config['data_used'])
    devices_data, devices_labels, server_test_features, server_test_labels = data_preparer.prepare_data()
    server_device = Device(idx=num_devices, data=server_test_features, labels=server_test_labels, maxiter=maxiter,
                           warm_start=True)
    device = Device(client_id, devices_data[client_id], devices_labels[client_id], maxiter=maxiter, warm_start=True)
    devices_list.append(device)
    print("[------------------DEVICES LIST----------------]", devices_list)

    threads_list = []
    port = peers[client_id]['port']
    threading.Thread(target=start_server, args=(client_id, port, threads_list)).start()

    initial_point = np.asarray([0.5] * ansatz.num_parameters)
    device.vqc.initial_point = initial_point

    if client_id == 0:
        train_events[0].set()

    for round_num in range(NUM_ROUNDS):
        print(
            f"[**************************START COMMUNICATION ROUND*********************** {round_num}] Starting round {round_num}")
        comm_start_time = time.time()

        print(f"Client {client_id} waiting for training signal for round {round_num}")
        train_events[client_id].wait()
        print(f"Client {client_id} received training signal for round {round_num}")

        # Wait for server training event
        # server_done_event.wait()/

        print(f"Client {client_id} starting training for round {round_num}...")

        device.training()
        with open(f"{logs_dir}/device.txt", 'a') as file:
            file.write(
                f"Comm_round: {round_num} - Device: {device.idx} - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f}\n")
        print(
            f"Comm_round: {round_num} - Device: {device.idx} - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f}")
        print(f"Device ID: {client_id} completed round {round_num}")
        weights = device.vqc.weights
        data_to_send = {'type': 'weights', 'client_id': client_id, 'weights': weights, 'round': round_num}

        next_peer = peers[(client_id + 1) % num_devices]
        send_to_peer(next_peer, data_to_send)
        print(f"Client {client_id} sent weights to next peer")

        train_events[client_id].clear()
        print(f"Client {client_id} waiting to receive weights for next round")

        while (client_id - 1) % num_devices not in peer_weights or peer_weights[(client_id - 1) % num_devices][
            0] != round_num:
            time.sleep(1)

        print(f"Client {client_id} received signal to proceed to round {round_num + 1}")

        received_data = peer_weights[(client_id - 1) % num_devices]
        if received_data and received_data[0] == round_num:
            device.vqc.initial_point = received_data[1]
            print(f"Client {client_id} received weights from previous peer")

        if client_id == num_devices - 1:
            server_time = time.time()
            server_training_event.clear()  # Clear the server training event before starting server thread
            server_thread(server_device, round_num, device.vqc.weights, logs_dir)
            server_done_event.wait()  # Wait for the server to complete
            server_time = time.time() - server_time
        else:
            server_time = 0

        # Make all devices wait for the server to complete
        # server_done_event.wait()

        comm_end_time = time.time() - comm_start_time - server_time
        with open(f"{logs_dir}/comm_time.txt", 'a') as file:
            file.write(f"Comm_round: {round_num} - Device : {client_id} -  Comm_time: {comm_end_time}\n")
        print(f"Communication time for round {round_num}: {comm_end_time} seconds")
        print(
            f"[**************************FINISH COMMUNICATION ROUND*********************** {round_num}] Starting round {round_num}")

        # Reset the server_done_event for the next round
        server_done_event.clear()

    stop_event.set()
    for thread in threads_list:
        thread.join()
    print(f"Number of active threads after completion: {threading.active_count()}")


    with open(f"{logs_dir}/objective_values_devices.txt", 'a') as file:
        for device in devices_list:
            file.write(f"Device {device.idx}: {device.objective_func_vals}\n")

    with open(f"{logs_dir}/server_objective_values_devices.txt", 'w') as file:
        # for device in devices_list:
        file.write(f"Device {server_device.idx}: {server_device.objective_func_vals}\n")


if __name__ == "__main__":
    main()
