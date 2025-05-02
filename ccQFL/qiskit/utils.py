import numpy as np

def shuffle_and_distribute_data(alldevices_train_features, alldevices_train_labels, num_devices, data_size, subset_size_device):
    if data_size == "small":
        total_samples = subset_size_device
    else:
        total_samples = len(alldevices_train_features)

    random_indices = np.random.choice(len(alldevices_train_features), total_samples, replace=False)
    alldevices_train_features = alldevices_train_features[random_indices]
    alldevices_train_labels = alldevices_train_labels[random_indices]

    samples_per_device = total_samples // num_devices
    remainder = total_samples % num_devices

    devices_data = []
    devices_labels = []
    start_index = 0
    for i in range(num_devices):
        extra_samples = 1 if i < remainder else 0
        end_index = start_index + samples_per_device + extra_samples
        device_data = np.array(alldevices_train_features[start_index:end_index])
        device_labels = np.array(alldevices_train_labels[start_index:end_index])
        start_index = end_index
        devices_data.append(device_data)
        devices_labels.append(device_labels)

    for i, (data, labels) in enumerate(zip(devices_data, devices_labels)):
        print(f"Device {i + 1} data:", data)
        print(f"Device {i + 1} labels:", labels)
        print()

    return devices_data, devices_labels