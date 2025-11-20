import tensorcircuit as tc
import jax.numpy as jnp
import optax
import jax
import numpy as np

# Configuration
no_of_qubits = 10
no_of_classes = 10
k = 42
readout_mode = 'softmax'
K = tc.set_backend('jax')

def filter(x, y, class_list):
    y_flat = y.flatten()
    keep = np.zeros(len(y_flat), dtype=bool)
    for c in class_list:
        keep |= (y_flat == c)
    x_filtered, y_filtered = x[keep], y_flat[keep]
    y_filtered = jax.nn.one_hot(y_filtered, no_of_classes)
    return x_filtered, y_filtered

def clf(params, c, k):
    print("CLF Function")
    for j in range(k):
        for i in range(no_of_qubits - 1):
            c.cnot(i, i + 1)
        for i in range(no_of_qubits):
            c.rx(i, theta=params[3 * j, i])
            c.rz(i, theta=params[3 * j + 1, i])
            c.rx(i, theta=params[3 * j + 2, i])
    return c

def readout(c):
    if readout_mode == 'softmax':
        logits = []
        for i in range(no_of_qubits):
            logits.append(jnp.real(c.expectation([tc.gates.z(), [i,]])))
        logits = jnp.stack(logits, axis=-1) * no_of_qubits
        probs = jax.nn.softmax(logits)
    elif readout_mode == 'sample':
        wf = jnp.abs(c.wavefunction()[:no_of_qubits])**2
        probs = wf / jnp.sum(wf)
    return probs

def loss(params, x, y, k):
    c = tc.Circuit(no_of_qubits, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-7), axis=-1))
loss = K.jit(loss, static_argnums=[3])

def accuracy(params, x, y, k):
    print("Accuracy Function")
    c = tc.Circuit(no_of_qubits, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return jnp.argmax(probs, axis=-1) == jnp.argmax(y, axis=-1)
accuracy = K.jit(accuracy, static_argnums=[3])

compute_loss = K.jit(K.vectorized_value_and_grad(loss, vectorized_argnums=[1, 2]), static_argnums=[3])
compute_accuracy = K.jit(K.vmap(accuracy, vectorized_argnums=[1, 2]), static_argnums=[3])

def pred(params, x, k):
    print("Pred Function")
    c = tc.Circuit(no_of_qubits, inputs=x)
    c = clf(params, c, k)
    probs = readout(c)
    return probs
pred = K.vmap(pred, vectorized_argnums=[1])

class Device:
    def __init__(self, id, data, params, opt_state):
        self.id = id
        self.data_train = data
        self.old_params = params
        self.params = params
        self.params_p = params
        self.euclidean_list = []
        self.params_list = []
        self.opt = optax.adam(learning_rate=1e-2)
        self.opt_state = opt_state
        self.sk = None
        self.params_hash = None
        self.pk = None
        self.train_list = []
        self.train_loss = []
        self.signature = None
        self.hash_signature = None