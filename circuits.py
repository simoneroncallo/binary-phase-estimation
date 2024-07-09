### This script prepares the quantum circuit that simulates the interferometric
### square-wave response. It consists in a sequence of beam splitters
### and (independent and parametric) phase transformations

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

def interferometric_module(p, p1, num_qubits: int, label = None):
    """ Prepare a single interferometric iteration, 
        with input and parametric phases p and p1, respectively. """
    qubits = QuantumRegister(num_qubits)
    circuit = QuantumCircuit(qubits)
    
    if p1 != 0:
        circuit.h(qubits)
        circuit.p(p1, qubits)
        circuit.h(qubits)
    circuit.p(p, qubits)
    
    return circuit.to_instruction(label = label)

def circuit_builder(in_phase, par_phases):
    """ Return a quantum circuit with square-wave response, obtained by 
        multiple subsequent applications of the interferometric module. """    
    num_qubits = 1 # Fixed
    num_bits = 1 # Fixed
    
    # Initialization
    q = QuantumRegister(num_qubits)
    c = ClassicalRegister(num_bits)
    qc = QuantumCircuit(q,c)

    # Preparation
    num_iters = len(par_phases) # Number of compositions
    for idx in range(num_iters):
        p1 = par_phases[idx] # Parametric phase
        qc.compose(interferometric_module(in_phase, 0, num_qubits,\
                          f'InMod#{idx}'), q, inplace = True)
        qc = qc.decompose(gates_to_decompose=[f'InMod#{idx}'])
        qc.compose(interferometric_module(in_phase, p1, num_qubits,\
                          f'InMod#{idx}'), q, inplace = True)
        qc = qc.decompose(gates_to_decompose=[f'InMod#{idx}'])          
    qc.h(q)
    
    # Measurement
    qc.measure(q, c)
    
    return qc