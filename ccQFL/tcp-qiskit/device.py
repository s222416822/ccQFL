

from qiskit_algorithms.utils import algorithm_globals
from sklearn.model_selection import train_test_split
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from matplotlib import pyplot as plt
from IPython.display import clear_output
import time
from qiskit_machine_learning.algorithms.classifiers import VQC
from sklearn.preprocessing import MinMaxScaler
from qiskit.primitives import Sampler

algorithm_globals.random_seed = 123

class Device:
    def __init__(self, idx, data, labels, maxiter=100, warm_start=None, initial_point=None):
        self.idx = idx
        self.features = MinMaxScaler().fit_transform(data)
        self.target = labels
        self.maxiter = maxiter
        self.train_score_q4 = 0
        self.test_score_q4 = 0
        self.training_time = 0
        self.sampler = Sampler()
        self.optimizer = COBYLA(maxiter=self.maxiter)
        self.objective_func_vals = []
        self.num_features = self.features.shape[1]
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(
            self.features, self.target, train_size=0.8, random_state=algorithm_globals.random_seed
        )
        self.feature_map = ZZFeatureMap(feature_dimension=self.num_features, reps=1)
        self.ansatz = RealAmplitudes(num_qubits=self.num_features, reps=3)
        self.warm_start = warm_start
        self.vqc = VQC(
            sampler=self.sampler,
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            callback=self.callback_graph,
            initial_point=initial_point,
            warm_start=self.warm_start
        )

    def get_data(self):
        return self.features

    def get_target(self):
        return self.target

    def set_data(self, data):
        self.features = MinMaxScaler().fit_transform(data)

    def set_target(self, target):
        self.target = target
    def callback_graph(self, weights, obj_func_eval):
        clear_output(wait=True)
        self.objective_func_vals.append(obj_func_eval)
        print(obj_func_eval)
        # plt.title(f"Device: {self.idx}")
        # plt.xlabel("Iter")
        # plt.ylabel("Loss")
        # plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals)
        # plt.show()

    def training(self):
        start = time.time()
        self.vqc.fit(self.train_features, self.train_labels)
        self.training_time = time.time() - start
        print(f"Training time: {round(self.training_time)} seconds")
        self.train_score_q4 = self.vqc.score(self.train_features, self.train_labels)
        self.test_score_q4 = self.vqc.score(self.test_features, self.test_labels)
        print(f"Quantum VQC on the training dataset: {self.train_score_q4:.2f}")
        print(f"Quantum VQC on the test dataset:     {self.test_score_q4:.2f}")

    def evaluate(self, weights):
        self.vqc.initial_point = weights
        self.test_score_q4_1 = self.vqc.score(self.test_features, self.test_labels)

