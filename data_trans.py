'''
Dependencies:
    qiskit
    qiskit-aer
    pylatexenc (maybe no need?)
    matplotlib
    pandas
    numpy
'''
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

def PCA(trainData, testData, nQubits):
    pass