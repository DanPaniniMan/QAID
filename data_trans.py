'''
Dependencies:
    qiskit
    qiskit-aer
    pylatexenc (maybe no need?)
    matplotlib
    pandas
    numpy
'''
# Libraries implied by the methodology (Scikit-learn, based on Algorithm 2)
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
# Let's go ahead and import all this stuff too
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
import numpy as np
from qiskit.visualization import plot_histogram, plot_state_qsphere, plot_bloch_multivector, plot_bloch_vector
#import getpass
import random
#import os
from IPython.display import clear_output
#import time

'''
@inputData -> vector states to be transformed
@nQubits -> the number of vector states to be transformed
Takes a number of qubit vector states and performs a double angle data 
transformation
'''
def doubleAngle(inputData, nQubits):
    outputData = pd.DataFrame()
    for sv in inputData['statevector']:    #change this with the appropriate thing for the dataframe
        for i in range(nQubits):
            pass
    return outputData


"""
trainData and testData are DataFrames
nQubits should be implemented with 7
This code was written with the help of Gemini 3
"""
def pca_data_transformation(trainData, testData, nQubits):
    """
    Algorithm 2: PCA Data Trans. (PCA)
    Reduces the state vector data to 14 principal components.
    """
    # The target dimension is 14 columns, derived from 7 qubits (2 values per qubit) [6].
    # Initialization of the PCA transformer
    n_components = nQubits * 2  # 7 * 2 = 14
    pca_transformer = PCA(n_components=n_components)
    
    # Fit the transformer using the training data
    pca_transformer.fit(trainData)
    
    # Transform both the training and testing data
    outputTrain = pca_transformer.transform(trainData)
    outputTest = pca_transformer.transform(testData)
    
    return pd.DataFrame(outputTrain), pd.DataFrame(outputTest)

#tests the PCA on random values
np.random.seed(42)
trainData = pd.DataFrame(np.random.randn(100, 128), columns=[f'col_{i}' for i in range(128)])
testData = pd.DataFrame(np.random.randn(100, 128), columns=trainData.columns)
nQubits = 7
outputTrain_pca, outputTest_pca = pca_data_transformation(trainData, testData, nQubits)
print(outputTrain_pca)

print(outputTest_pca)
# The output data will have 14 columns, maintaining maximum variance [7].