import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

# Configuration
num_classes = 3
feature_size = 4
margin = 0.15
num_layers = 6
batch_size = 10
num_qubits = int(np.ceil(np.log2(feature_size)))

class Device:
    def __init__(self, idx, feats_train, y_train, feats_val, y_val, num_train, features, Y, simulator):
        self.idx = idx
        if simulator == "qiskit.remote":
            backend = FakeManilaV2()
            self.dev = qml.device("qiskit.remote", wires=5, backend=backend)
        elif simulator == "qiskit.basicsim":
            self.dev = qml.device("qiskit.basicsim", wires=2)
        elif simulator == "qiskit.aer":
            self.dev = qml.device("qiskit.aer", wires=2)
        self.feat_vecs_train = feats_train
        self.feat_vecs_test = feats_val
        self.feats_train = feats_train
        self.Y_train = y_train
        self.feats_val = feats_val
        self.Y_test = y_val
        self.y_val = y_val
        self.qnodes = []
        for iq in range(num_classes):
            qnode = qml.QNode(circuit, self.dev, interface="torch")
            self.qnodes.append(qnode)
        self.q_circuits = self.qnodes

        # Initialize the parameters
        self.weights = [
            Variable(0.1 * torch.randn(num_layers, num_qubits, 3), requires_grad=True)
            for i in range(num_classes)
        ]
        self.biases = [Variable(0.1 * torch.ones(1), requires_grad=True) for i in range(num_classes)]
        self.optimizer = optim.Adam(self.weights + self.biases, lr=0.01)
        self.params = (self.weights, self.biases)
        self.params_p = self.params
        self.old_weights = self.weights
        self.num_train = num_train
        self.features = features
        self.Y = Y

def layer(W):
    for i in range(num_qubits):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
    for j in range(num_qubits - 1):
        qml.CNOT(wires=[j, j + 1])
    if num_qubits >= 2:
        # Apply additional CNOT to entangle the last with the first qubit
        qml.CNOT(wires=[num_qubits - 1, 0])

def circuit(weights, feat=None):
    qml.AmplitudeEmbedding(feat, range(num_qubits), pad_with=0.0, normalize=True)
    for W in weights:
        layer(W)
    return qml.expval(qml.PauliZ(0))

def multiclass_svm_loss(q_circuits, all_params, feature_vecs, true_labels):
    loss = 0
    num_samples = len(true_labels)
    for i, feature_vec in enumerate(feature_vecs):
        s_true = variational_classifier(
            q_circuits[int(true_labels[i])],
            (all_params[0][int(true_labels[i])], all_params[1][int(true_labels[i])]),
            feature_vec,
        )
        s_true = s_true.float()
        li = 0
        for j in range(num_classes):
            if j != int(true_labels[i]):
                s_j = variational_classifier(
                    q_circuits[j], (all_params[0][j], all_params[1][j]), feature_vec
                )
                s_j = s_j.float()
                li += torch.max(torch.zeros(1).float(), s_j - s_true + margin)
        loss += li
    return loss / num_samples

def classify(q_circuits, all_params, feature_vecs, labels):
    predicted_labels = []
    for i, feature_vec in enumerate(feature_vecs):
        scores = np.zeros(num_classes)
        for c in range(num_classes):
            score = variational_classifier(
                q_circuits[c], (all_params[0][c], all_params[1][c]), feature_vec
            )
            scores[c] = float(score)
        pred_class = np.argmax(scores)
        predicted_labels.append(pred_class)
    return predicted_labels

def accuracy(labels, hard_predictions):
    loss = 0
    for l, p in zip(labels, hard_predictions):
        if torch.abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / labels.shape[0]
    return loss

def variational_classifier(q_circuit, params, feat):
    weights = params[0]
    bias = params[1]
    return q_circuit(weights, feat=feat) + bias



# def cost(weights, bias, X, Y):
#   predictions = [variational_classifier(weights, bias, x) for x in X]
#   return square_loss(Y, predictions)


def cost(weights, bias, X, Y):
    # Transpose the batch of input data in order to make the indexing
    # in state_preparation work
    predictions = variational_classifier(weights, bias, X.T)
    return square_loss(Y, predictions)

def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(np.linalg.norm(x[2:]) / np.linalg.norm(x))

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


def state_preparation(a):
    qml.RY(a[0], wires=0)

    qml.CNOT(wires=[0, 1])
    qml.RY(a[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[2], wires=1)

    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[4], wires=1)
    qml.PauliX(wires=0)