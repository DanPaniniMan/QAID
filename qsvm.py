import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
import time

from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.primitives import Sampler

#parameters to adjust
model_name = "qsvc_model_3000.pkl"
train_size = 3000
FEATURE_COUNT = 4 # creates a 4 qubit system

#load and split data
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

X = train_df.drop('label', axis=1)
y = train_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= int(train_size/4), train_size=train_size, random_state=100)

# --- Process the dataframes for initializing the Quantum Kernel ---

pca = PCA(n_components=FEATURE_COUNT)
scaler = MinMaxScaler((0, 2 * np.pi)) #scales to valid angle values

pca.fit(X_train)
X_train_features = pca.transform(X_train) #creates 4 features for 4 qubits

scaler.fit(X_train_features)
X_train_scaled_features = scaler.transform(X_train_features) #scale to rotations

X_test_features = pca.transform(X_test)
X_test_scaled_features = scaler.transform(X_test_features)

# --- create the feature map to embed our data into ---
# qiskit docs source: https://qiskit-community.github.io/qiskit-machine-learning/tutorials/03_quantum_kernel.html
feature_map = ZZFeatureMap(
    feature_dimension=FEATURE_COUNT, 
    reps=2, 
    entanglement='linear' # Connects adjacent qubits
)

# for simulation
sampler = Sampler()

# computes fidelity
fidelity = ComputeUncompute(sampler=sampler)

adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# --- Construct and train the quantum kernel ---
qsvc = QSVC(quantum_kernel=adhoc_kernel) #QSVC model creation

print("kernel initialized, now training")
start = time.time()

qsvc.fit(X_train_scaled_features, y_train) #finds the hyperplane

end = time.time()
train_time = end - start

print("time to train: " + str(train_time))

qsvc.save(model_name)

print("finished training, now scoring")

qsvc_score = qsvc.score(X_test_scaled_features, y_test)
print(f"QSVC classification test score: {qsvc_score}")

# code for loading a model to test
# loaded_qsvc_model = QSVC.load(model_name)
# Now you can use the loaded_model for prediction or further training
# For example:
# predictions = loaded_qsvc_model.predict(X_test_scaled_features)
# print(predictions)
