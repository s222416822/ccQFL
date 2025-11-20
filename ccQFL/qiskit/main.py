from datetime import datetime
import os
from data_loader import load_and_preprocess_data
from device import Device
from utils import shuffle_and_distribute_data
from qiskit.primitives import Sampler as Sampler1
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
import numpy as np

data_used = "iris"  # Options: synthetic, iris, genomics, mnist, mnist_keras, fashion

data_size = "normal"  # Options: normal, small
# data_size = "small"
subset_size_device = 400
subset_size_server = 50
pca_n_components = [4, 10, 20, 40, 100]
random_number = 30
maxiter = "30" if random_number == 30 else "random"

# Initialize Qiskit Runtime Service
service = QiskitRuntimeService(
    channel="ibm_quantum",
    token="YOUR IBM TOKEN"
)

def main_method(algorithm, optimizer, pca_n_component, simulator, sampler, aer_sim):
    # Load and preprocess data
    (
        alldevices_train_features,
        server_test_features,
        alldevices_train_labels,
        server_test_labels,
        num_devices
    ) = load_and_preprocess_data(data_used, data_size, subset_size_device, subset_size_server)

    # Shuffle and distribute data
    devices_data, devices_labels = shuffle_and_distribute_data(
        alldevices_train_features,
        alldevices_train_labels,
        num_devices,
        data_size,
        subset_size_device
    )

    # Subset server data if small
    if data_size == "small":
        server_test_features = server_test_features[:subset_size_server]
        server_test_labels = server_test_labels[:subset_size_server]

    # Create devices
    devices_list = []
    for i in range(num_devices):
        device = Device(
            idx=i,
            data=devices_data[i],
            labels=devices_labels[i],
            optimizer=optimizer,
            pca_n_component=pca_n_component,
            simulator=simulator,
            sampler_object=sampler,
            aer_sim=aer_sim,
            maxiter=random_number,
            warm_start=True
        )
        devices_list.append(device)

    server_device = Device(
        idx=num_devices,
        data=server_test_features,
        labels=server_test_labels,
        optimizer=optimizer,
        pca_n_component=pca_n_component,
        simulator=simulator,
        sampler_object=sampler,
        aer_sim=aer_sim,
        maxiter=random_number,
        warm_start=True
    )

    # Setup logging
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    logs = f"logs/{simulator}_{data_used}/{algorithm}_{optimizer}_{pca_n_component}_{date_time}_{data_used}_{subset_size_device}_{subset_size_server}_maxiter={maxiter}_numDevices={num_devices}"
    if not os.path.exists(logs):
        os.makedirs(logs)

    # Algorithm implementations
    if algorithm == "defaultQFL":
        average_weights = None
        for n in range(1):
            import time
            comm_start_time = time.time()
            total_weights = []
            for device in devices_list:
                device.current_comm_round = n
                if n == 0:
                    # Initialize weights for the first round.
                    device.vqc.initial_point = np.asarray([0.5] * device.ansatz.num_parameters)
                else:
                    # Use average weights for subsequent rounds.
                    device.vqc.initial_point = average_weights
                print(f"Device {device.idx} is training...")
                device.training()
                with open(f"{logs}/device_params.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device {device.idx} - params: {device.vqc.weights}\n")
                print("VQC fit Results", device.vqc.fit_result)
                print(f"Device {device.idx} successfully completed processing.")
                total_weights.append(device.vqc.weights)
                with open(f"{logs}/device.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device: {device.idx}  - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f}\n")
                with open(f"{logs}/training_time_device.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device: {device.idx} - training_time: {device.training_time}\n")

            average_weights = np.mean(total_weights, axis=0)
            with open(f"{logs}/average_weights.txt", 'a') as file:
                file.write(f"Comm_round: {n} - average_weights: {average_weights}\n")

            comm_end_time = time.time() - comm_start_time
            print(f"Comm_round: {n} - Comm_time: {comm_end_time}")
            with open(f"{logs}/comm_time.txt", 'a') as file:
                file.write(f"Comm_round: {n} - Comm_time: {comm_end_time}\n")
        with open(f"{logs}/objective_values_devices.txt", 'w') as file:
            for device in devices_list:
                file.write(f"Device {device.idx}: {device.objective_func_vals}\n")

        with open(f"{logs}/device_params_per_iter.txt", 'w') as file:
            for device in devices_list:
                file.write(f"Device {device.idx}: {device.params_per_iter}\n")


    elif algorithm == "optimized-defaultQFL":
        average_weights = None
        for n in range(10):
            import time
            import threading
            comm_start_time = time.time()
            def train_device(device, n):
                if n == 0:
                    device.vqc.initial_point = np.asarray([0.5] * device.ansatz.num_parameters)
                else:
                    device.vqc.initial_point = average_weights
                print(f"Device {device.idx} is training...")
                device.training()
                with open(f"{logs}/device_params.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device {device.idx} - params: {device.vqc.weights}\n")
                with open(f"{logs}/device.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device: {device.idx}  - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f}\n")
                with open(f"{logs}/training_time_device.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device: {device.idx} - training_time: {device.training_time}\n")
            threads_train_device = []
            for device in devices_list:
                device.current_comm_round = n
                device_thread = threading.Thread(target=train_device, args=(device, n))
                device_thread.start()
                threads_train_device.append(device_thread)
            for thread in threads_train_device:
                thread.join()
            print("Finished all devices training...")
            import numpy as np
            weights_list = [device.vqc.weights for device in devices_list]
            average_weights = np.mean(weights_list, axis=0)
            with open(f"{logs}/average_weights.txt", 'a') as file:
                file.write(f"Comm_round: {n} - average_weights: {average_weights}\n")

            comm_end_time = time.time() - comm_start_time
            print(f"Comm_round: {n} - Comm_time: {comm_end_time}")
            with open(f"{logs}/comm_time.txt", 'a') as file:
                file.write(f"Comm_round: {n} - Comm_time: {comm_end_time}\n")
        with open(f"{logs}/objective_values_devices.txt", 'w') as file:
            for device in devices_list:
                file.write(f"Device {device.idx}: {device.objective_func_vals}\n")

        with open(f"{logs}/device_params_per_iter.txt", 'w') as file:
            for device in devices_list:
                file.write(f"Device {device.idx}: {device.params_per_iter}\n")


    elif algorithm == "optimized-chainedQFL":
        import time

        def shuffle():
            indices = list(range(len(devices_list)))
            random.shuffle(indices)
            for i in indices:
                print(devices_list[i].idx)
        weights_chained = []
        last_device_id = 0
        for n in range(10):
            # shuffle()
            comm_start_time = time.time()
            def process_devices(devicess_list, n):
                round_num = 0
                while len(devicess_list) > 3:
                    print(f"\nRound {round_num}:")
                    paired_devices = []
                    first_devices = []
                    second_devices = []
                    for i in range(0, len(devicess_list) - 1, 2):
                        pair = (devicess_list[i], devicess_list[i + 1])
                        paired_devices.append(pair)
                        first_devices.append(pair[0])
                        second_devices.append(pair[1])
                    if len(devicess_list) % 2 != 0:
                        unpaired_device = devicess_list[-1]
                        print(f"Unpaired Device: {unpaired_device}")
                        second_devices.append(unpaired_device)
                    print(f"Paired Devices: {paired_devices}")
                    print(f"First Devices from Pairs: {first_devices}")
                    threads_train_device = []
                    for device in first_devices:
                        device.current_comm_round = n
                        pre_weights_chained_list = []
                        print(f"Comm: {n} - Round Num: {round_num} - Devices to train: [First Devices] {[device.idx for device in first_devices]}")
                        if round_num < 1:
                            if n < 1:
                                device_thread = threading.Thread(target=train_device, args=(device, n))
                                device_thread.start()
                                threads_train_device.append(device_thread)
                                print("Finished all devices training...")
                            else:
                                device_thread = threading.Thread(target=train_device, args=(device, n, None, weights_chained))
                                device_thread.start()
                                threads_train_device.append(device_thread)
                                print("Finished all devices training...")
                        else:
                            if round_num == 1:
                                id = device.idx
                                weights = devices_list[id-1].vqc.weights
                                pre_weights_chained_list.append(weights)
                            elif round_num > 1:
                                id = device.idx
                                weights1 = devices_list[id-1].vqc.weights
                                weights2 = devices_list[id - (2 ** (round_num - 1))].vqc.weights
                                pre_weights_chained_list.append(weights1)
                                pre_weights_chained_list.append(weights2)
                            device_thread = threading.Thread(target=train_device, args=(device, n, pre_weights_chained_list, None))
                            device_thread.start()
                            threads_train_device.append(device_thread)
                    for thread in threads_train_device:
                        thread.join()
                    devicess_list = second_devices
                    round_num += 1
                print(f"\nRemaining Devices: {devicess_list} (Less than 4, cannot pair further)")
                pre_weights = None
                for device in devicess_list:
                    device.current_comm_round = n
                    pre_weights_chained_last = []
                    print(f"Comm: {n} - Round Num: {round_num} - Devices to train: [REMAINING] {[device.idx for device in devicess_list]}")
                    id = device.idx
                    if round_num == 1:
                        weights = devices_list[id-1].vqc.weights
                        pre_weights_chained_last.append(weights)
                    elif round_num > 1:
                        print("Device id on remaining devices;", id)
                        if id == devicess_list[0].idx:
                            weights1 = devices_list[id-1].vqc.weights
                            weights2 = devices_list[id - (2 ** (round_num - 1))].vqc.weights
                            pre_weights_chained_last.append(weights1)
                            pre_weights_chained_last.append(weights2)
                        else:
                            print(f"Device last remaining: {device.idx}: Pre Weights = {pre_weights}")
                            weights1 = devices_list[id-1].vqc.weights
                            weights2 = devices_list[id - (2 ** (round_num - 2))].vqc.weights
                            weights3 = pre_weights
                            pre_weights_chained_last.append(weights1)
                            pre_weights_chained_last.append(weights2)
                            pre_weights_chained_last.append(weights3)
                    last_device_id = id
                    train_device(device, n, pre_weights_chained_last, None)
                    pre_weights = device.vqc.weights
            def train_device(device, n, pre_weights_chained_list=None, weights_chained=None):
                if n == 0:
                    device.vqc.initial_point = np.asarray([0.5] * device.ansatz.num_parameters)
                else:
                    if pre_weights_chained_list:
                        weights_chained_for_device = np.mean(pre_weights_chained_list, axis=0)
                        device.vqc.initial_point = weights_chained_for_device
                    else:
                        device.vqc.initial_point = weights_chained

                # Perform training for the current device.
                print(f"Training Device {device.idx}...")
                device.training()
                with open(f"{logs}/device_params.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device {device.idx} - params: {device.vqc.weights}\n")
                print(f"Device {device.idx} successfully completed processing.")
                with open(f"{logs}/training_time_device.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device: {device.idx} - training_time: {device.training_time}\n")
                with open(f"{logs}/device.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device: {device.idx}  - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f}\n")

            process_devices(devices_list, n)
            print(f"Last device ID: {last_device_id} - Weights Chained: {weights_chained}")
            weights_chained = devices_list[last_device_id].vqc.weights
            with open(f"{logs}/last_device_chained_weights.txt", 'a') as file:
                file.write(f"Comm_round: {n} - average_weights: {weights_chained}\n")
            comm_end_time = time.time() - comm_start_time
            print(f"Comm_round: {n} - Comm_time: {comm_end_time}")
            with open(f"{logs}/comm_time.txt", 'a') as file:
                file.write(f"Comm_round: {n} - Comm_time: {comm_end_time}\n")
            with open(f"{logs}/chained_weights.txt", 'a') as file:
                file.write(f"Comm_round: {n} - average_weights: {weights_chained}\n")

        with open(f"{logs}/objective_values_devices.txt", 'w') as file:
            for device in devices_list:
                file.write(f"Device {device.idx}: {device.objective_func_vals}\n")

        with open(f"{logs}/device_params_per_iter.txt", 'w') as file:
            for device in devices_list:
                file.write(f"Device {device.idx}: {device.params_per_iter}\n")


    elif algorithm == "chainedQFL":
        import random
        import numpy as np
        import time
        def shuffle():
            indices = list(range(len(devices_list)))
            random.shuffle(indices)
            for i in indices:
                print(devices_list[i].idx)
        weights_chained = None
        for n in range(1):
            comm_start_time = time.time()
            last_online_index = None
            for i, device in enumerate(devices_list):
                device.current_comm_round = n
                # Step 3: Handle the first device (i == 0) and first round (n == 0).
                if n == 0 and i == 0:
                    device.vqc.initial_point = np.asarray([0.5] * device.ansatz.num_parameters)
                else:
                    device.vqc.initial_point = weights_chained
                print(f"Training Device {device.idx}...")

                device.training()
                with open(f"{logs}/device_params.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device {device.idx} - params: {device.vqc.weights}\n")
                weights_chained = device.vqc.weights
                print(f"Device {device.idx} successfully completed processing.")
                with open(f"{logs}/training_time_device.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device: {device.idx} - training_time: {device.training_time}\n")
                with open(f"{logs}/device.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device: {device.idx}  - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f}\n")
            comm_end_time = time.time() - comm_start_time
            print(f"Comm_round: {n} - Comm_time: {comm_end_time}")
            with open(f"{logs}/comm_time.txt", 'a') as file:
                file.write(f"Comm_round: {n} - Comm_time: {comm_end_time}\n")
            with open(f"{logs}/chained_weights.txt", 'a') as file:
                file.write(f"Comm_round: {n} - average_weights: {weights_chained}\n")

        with open(f"{logs}/objective_values_devices.txt", 'w') as file:
            for device in devices_list:
                file.write(f"Device {device.idx}: {device.objective_func_vals}\n")

        with open(f"{logs}/device_params_per_iter.txt", 'w') as file:
            for device in devices_list:
                file.write(f"Device {device.idx}: {device.params_per_iter}\n")


if __name__ == "__main__":
    algorithms = [
        'defaultQFL',
        'optimized-defaultQFL',
        'optimized-chainedQFL',
        'chainedQFL'
    ]

    # optimizers = ['adam', 'aqgd', 'cobyla', 'gradientdescent', 'powell', 'qnspsa']

    simulators = [
        # 'sampler',
        'aer_sim',
        'aer_sim_ibm_brisbane',
        'fake_manila',
        # 'clifford'
        # 'real_quantum'
    ]
    aer_sim = None
    sampler = None
    for simulator in simulators:
        if simulator == "sampler":
            sampler = Sampler1()
            aer_sim = AerSimulator()
        elif simulator == "aer_sim":
            aer_sim = AerSimulator()
            sampler = Sampler(mode=aer_sim)
        elif simulator == "aer_sim_ibm_brisbane":
            # Specify a QPU to use for the noise model
            real_backend = service.backend("ibm_brisbane")
            aer_sim = AerSimulator.from_backend(real_backend)
            # aer_sim = AerSimulator(noise_model=noise_model)
            sampler = Sampler(mode=aer_sim)
        elif simulator == "fake_manila":
            fake_manila = FakeManilaV2()
            sampler = Sampler(mode=fake_manila)
            aer_sim = fake_manila
        # elif simulator == "clifford":
        #   aer_sim = AerSimulator(method="stabilizer")
        #   sampler = Sampler(mode=aer_sim)
        # elif simulator == "real_quantum":
        #   # backend = service.backend('ibm_quantum')
        #   backend = service.least_busy(operational=True, simulator=False)
        #   aer_sim = backend
        # sampler = Sampler(mode=aer_sim)

        for algorithm in algorithms:
            print(f"Algorithm: {algorithm}, Simulator: {simulator}")
            main_method(
                algorithm=algorithm,
                optimizer="cobyla",
                pca_n_component=4,
                simulator=simulator,
                sampler=sampler,
                aer_sim=aer_sim
            )