# Chained continuous quantum federated learning framework
Future Generation Computer Systems

This repository contains the code used for the paper 
titled **"Chained continuous quantum federated learning framework"**. 

<p style="text-align: center;">
  <img src="/ccQFL/images/ccsQFL.png" width="400" height="">
</p>


## Abstract
The integration of quantum machine learning into federated learning paradigms is poised to transform the future of technologies 
that depend on diverse machine learning methodologies. This research delves into Quantum Federated Learning (QFL), 
presenting an initial framework modeled on the Federated Averaging (FedAvg) algorithm, implemented via Qiskit. 
Despite its potential, QFL encounters critical challenges, including (i) susceptibility to a single point of failure, 
(ii) communication bottlenecks, and (iii) uncertainty in model convergence. Subsequently, we dive deeper into QFL and 
propose an innovative alternative to traditional server-based QFL. Our approach introduces a chained continuous QFL framework (ccQFL), 
which eliminates the need for a central server and the FedAvg method. In our framework, clients engage in a 
chained continuous training process, where they exchange models and collaboratively enhance each other’s performance. 
This approach improves both the efficiency of communication and the accuracy of the training process. 
Our experimental evaluation includes a proof-of-concept to demonstrate initial feasibility and a 
prototype study simulating TCP/IP communication between clients. This simulation enables concurrent operations, 
verifying the potential of ccQFL for real-world applications. We examine various datasets, 
including Iris, MNIST, synthetic and Genomic, covering a range of data sizes from small to large. 
For further validity of our proposed method, we extend our experimental analysis in other frameworks 
such as PennyLane and TensorCircuit where we include various ablation studies covering major 
considerations and factors that impact the framework to study validity, robustness, practicality, and others. 
Our results show that the ccQFL framework achieves model convergence, and we evaluate other critical 
metrics such as performance and communication delay. In addition, we provide a theoretical analysis 
to establish and discuss many factors such as model convergence, communication costs, etc.

## Highlights
- QFL framework with Qiskit, including distributed learning via TCP/IP. 
- Chained Continuous QFL (ccQFL) Framework that operates without aggregation. 
- Detailed assessment of ccQFL Framework, compared with standard QFL thoroughly. 
- Experiments extended to PennyLane, TensorCircuit, validated in many scenarios.

## Paper
- **Journal Version**: Future Generation Computer Systems [https://doi.org/10.1016/j.future.2025.107800]()

## Installation
- Qiskit:
  - pip install qiskit qiskit qiskit_algorithms qiskit_machine_learning pylatexenc genomic-benchmarks qiskit-aer
  - pip install qiskit_ibm_runtime

    
References:
1. https://github.com/Qiskit/qiskit
2. Ville Bergholm et al. PennyLane: Automatic differentiation of hybrid quantum-classical computations. 2018. arXiv:1811.04968
2. Zhao, H. (2023). Non-IID quantum federated learning with one-shot communication complexity. Quantum Machine Intelligence, 5(1), 3.
2. https://tensorcircuit.readthedocs.io/en/latest/
3. https://github.com/PennyLaneAI/qml

