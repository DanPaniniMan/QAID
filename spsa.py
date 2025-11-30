import numpy as np
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import pauli_two_design
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from lambeq import BobcatParser, IQPAnsatz, AtomicType, Rewriter
from discopy import monoidal

# from lambeq.rewrite import (
#     DiagramlyRewriter,
#     RuleRewriter,
#     DetACH,
#     Anaphora,
#     Preadverb,
#     Postadverb
# )


# ansatz = pauli_two_design(2, reps=1, seed=2)
# observable = SparsePauliOp("ZZ")
# initial_point = np.random.random(ansatz.num_parameters)
# estimator = StatevectorEstimator()
# def loss(x):
#     job = estimator.run([(ansatz, observable, x)])
#     return job.result()[0].data.evs



# spsa = SPSA(maxiter=300)
# result = spsa.minimize(loss, x0=initial_point)
# print(result)

def get_values(parser, rewriter, ansatz_obj, sentence: str):
    # ansatz = 
    return pauli_two_design(2, reps=1, seed=2)



# --- 1. initialize the parser ---
parser = BobcatParser()

# --- 2. Defining the Qubit Mapping and Creating the IQPAnsatz ---

# Define how many qubits each AtomicType (n, s, etc.) in the string diagram 
# will be mapped to.
# AtomicType.NOUN (n) and AtomicType.SENTENCE (s) are common types.
qubit_map = {
    AtomicType.NOUN: 1,      # Map noun wires (n) to 1 qubit
    AtomicType.SENTENCE: 1   # Map sentence wires (s) to 1 qubit
}

# Instantiate the IQPAnsatz
ansatz_obj = IQPAnsatz(
    qubit_map,
    n_layers=1,             # Number of IQP layers
    n_single_qubit_params=3 # Number of single-qubit rotation parameters (Rx, Rz, Rx)
)

# example diagram
diagram = parser.sentence2diagram("John walks in the park")

# --- 3. Initializing rewriter wit the rules in the paper ---
# Apply determiner
rewriter = Rewriter(['determiner'])
rewritten_diagram = rewriter(diagram)

rewritten_diagram.draw(figsize=(11,5), fontsize=13)

ansatz = ansatz_obj(rewritten_diagram)

# # --- 3. Initializing rewriter wit the rules in the paper (with aid from gemini) ---
# """
# Requires the following rules:
# - The determiner, preadverb, and post-adverb rule
# - The cups in thediagrams are removed using the bigraph method (may require
# restructuring, such as moving
# all the cups below all the word boxes and ordering them
# such that all the cups on the right of a cup are positionedabove it)
# """
# linguistic_rule = RuleRewriter(
#     [
#         DetACH(),    # Detacher, preadverb, and post-adverb rule. DetACH covers determiners.
#         Preadverb(), # Preadverb rule
#         Postadverb() # Post-adverb rule
#     ]
# )

# cup_removal_rule = DiagramlyRewriter()

# # 3. Combine them into a single pipeline
# # 3. Combine them into a single pipeline
# rewriter_pipeline = [
#     linguistic_rule,
#     cup_removal_rule
# ]
# print(rewriter_pipeline) # You can print the pipeline to see its components