import random
import numpy as np
import time
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA, GradientDescent, ADAM, AQGD, POWELL, QNSPSA
from qiskit.primitives import Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_machine_learning.algorithms.classifiers import VQC
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals
import matplotlib.pyplot as plt

class Device:
    def __init__(self, idx, data, labels, optimizer, pca_n_component, simulator, aer_sim, sampler_object, maxiter=30, warm_start=None, initial_point=None):
        self.idx = idx
        self.features_encoded_pca = PCA(n_components=pca_n_component).fit_transform(data)
        self.features = MinMaxScaler().fit_transform(self.features_encoded_pca)
        self.target = labels
        self.maxiter = maxiter
        self.online = self.check_online()
        self.failure = self.check_failure()
        self.train_score_q4 = 0
        self.test_score_q4 = 0
        self.training_time = 0
        self.state = "online_working"
        self.current_comm_round = 0
        self.params_per_iter = []
        self.sampler = sampler_object
        self.aer_sim = aer_sim

        if optimizer == "cobyla":
            self.optimizer = COBYLA(maxiter=self.maxiter)
        elif optimizer == "gradientdescent":
            self.optimizer = GradientDescent(maxiter=self.maxiter)
        elif optimizer == "adam":
            self.optimizer = ADAM(maxiter=self.maxiter)
        elif optimizer == "powell":
            self.optimizer = POWELL(maxiter=self.maxiter)
        elif optimizer == 'qnspsa':
            self.optimizer = QNSPSA(maxiter=self.maxiter)
        elif optimizer == 'aqgd':
            self.optimizer = AQGD(maxiter=self.maxiter)

        self.objective_func_vals = []
        self.num_features = self.features.shape[1]
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(
            self.features, self.target, train_size=0.8, random_state=algorithm_globals.random_seed
        )
        self.feature_map = ZZFeatureMap(feature_dimension=self.num_features, reps=1)
        self.ansatz = RealAmplitudes(num_qubits=self.num_features, reps=3)
        self.warm_start = warm_start
        # self.vqc = VQC(
        #     sampler=self.sampler,
        #     feature_map=self.feature_map,
        #     ansatz=self.ansatz,
        #     optimizer=self.optimizer,
        #     callback=self.callback_graph,
        #     # initial_point=initial_point,
        #     warm_start=self.warm_start
        # )
        pm = generate_preset_pass_manager(backend=self.aer_sim, optimization_level=1)
        self.isa_qc_ansatz = pm.run(self.ansatz)
        self.isa_qc_feature_map = pm.run(self.feature_map)
        self.vqc = VQC(
            sampler=self.sampler,
            feature_map=self.isa_qc_feature_map,
            ansatz=self.isa_qc_ansatz,
            optimizer=self.optimizer,
            callback=self.callback_graph,
            warm_start=self.warm_start
        )
        self.transition_matrix = {
            "online_working": {
                "online_working": 0.7,
                "online_failed": 0.1,
                "offline_working": 0.15,
                "offline_failed": 0.05
            },
            "online_failed": {
                "online_working": 0.2,
                "online_failed": 0.6,
                "offline_working": 0.1,
                "offline_failed": 0.1
            },
            "offline_working": {
                "online_working": 0.5,
                "online_failed": 0.05,
                "offline_working": 0.4,
                "offline_failed": 0.05
            },
            "offline_failed": {
                "online_working": 0.2,
                "online_failed": 0.1,
                "offline_working": 0.2,
                "offline_failed": 0.5
            }
        }

    def transition_state(self, transition_matrix):
        """Transitions the device state based on the Markov Chain."""
        states, probs = zip(*transition_matrix[self.state].items())
        self.state = np.random.choice(states, p=probs)
        print(f"Device {self.idx} transitioned to {self.state}")

    def is_online(self):
        """Check if the device is online."""
        return "online" in self.state

    def is_failed(self):
        """Check if the device is in a failed state."""
        return "failed" in self.state

    def check_failure(self, prob=0.05):
        """
        Parameters:
        - failure_probability (float): Probability that the device will fail.
                                      Default is 0.1 (10% chance of failure).
        Returns:
        - bool: True if the device fails, False if it succeeds.
        """
        rand_value = random.random()
        print(f"Device {self.idx} rand_value: {rand_value}")
        if rand_value < prob:
            return True
        else:
            return False

    def check_online(self, probability=0.8):
        """
        Parameters:
        - online_probability (float): Probability that the device is online.
                                    Default is 0.8 (80% chance of being online).
        Returns:
        - bool: True if the device is online, False if offline.
        """
        rand_value = random.random()
        if rand_value < probability:
            return True
        else:
            return False

    def get_data(self):
        return self.features

    def get_target(self):
        return self.target

    def set_data(self, data):
        self.features = MinMaxScaler().fit_transform(data)

    def set_target(self, target):
        self.target = target

    def callback_graph(self, weights, obj_func_eval):
        # clear_output(wait=True)
        self.objective_func_vals.append(obj_func_eval)
        self.params_per_iter.append(weights)
        plt.title(f"Device: {self.idx}")
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals)
        print(f"Comm Round: {self.current_comm_round} - Device {self.idx} - Weights: {weights}\n")
        print(f"Comm Round: {self.current_comm_round} - Device {self.idx} - Objective Func Eval: {obj_func_eval}\n")
        # plt.show()

    def training(self, initial_point=None):
        print(f"Train Features Shape: {self.train_features.shape}")
        print(f"Train Labels Shape: {self.train_labels.shape}")
        print(f"First Few Labels:\n{self.train_labels[:5]}")
        start = time.time()
        self.vqc.fit(self.train_features, self.train_labels)
        self.training_time = time.time() - start
        print(f"Training time: {round(self.training_time)} seconds")
        self.train_score_q4 = self.vqc.score(self.train_features, self.train_labels)
        self.test_score_q4 = self.vqc.score(self.test_features, self.test_labels)
        print(f"Quantum VQC on the training dataset: {self.train_score_q4:.2f}")
        print(f"Quantum VQC on the test dataset:     {self.test_score_q4:.2f}")

    def log_status(self, n, device, status, logs):
        """Logs the online/offline status and whether the device failed."""
        with open(f"{logs}/device_status.txt", 'a') as file:
            file.write(f"Comm_round: {n} - Device: {device.idx} - Status: {status}\n")

    def evaluate(self, weights):
        self.vqc.initial_point = weights
        self.test_score_q4_1 = self.vqc.score(self.test_features, self.test_labels)