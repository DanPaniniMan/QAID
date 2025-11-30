import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.primitives import Sampler
# from qiskit_machine_learning.kernels import QuantumKernel

#load and split data
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

X = train_df.drop('label', axis=1)
y = train_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=100)

# --- Process the dataframes for initializing the Quantum Kernel ---
FEATURE_COUNT = 4 # creates a 4 qubit system
pca = PCA(n_components=FEATURE_COUNT)
scaler = MinMaxScaler((0, 2 * np.pi)) #scales to valid angle values

pca.fit(X_train)
X_train_features = pca.transform(X_train) #creates 4 features for 4 qubits

scaler.fit(X_train_features)
X_train_scaled_features = scaler.transform(X_train_features) #scale to rotations

X_test_features = pca.transform(X_test)
X_test_scaled_features = scaler.transform(X_test_features)

# create the feature map to embed our data into
# qiskit docs source: https://qiskit-community.github.io/qiskit-machine-learning/tutorials/03_quantum_kernel.html
feature_map = ZZFeatureMap(
    feature_dimension=FEATURE_COUNT, 
    reps=2, 
    entanglement='linear' # Connects adjacent qubits
)

sampler = Sampler()

fidelity = ComputeUncompute(sampler=sampler)

adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

qsvc = QSVC(quantum_kernel=adhoc_kernel)
print("kernel initialized, now training")
qsvc.fit(X_train_scaled_features, y_train)
print("finished training, now scoring")
qsvc_score = qsvc.score(X_test_features, y_test)

print(f"QSVC classification test score: {qsvc_score}")
