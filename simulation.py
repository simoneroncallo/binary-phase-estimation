### This script performs the simulation of the square-wave interferometer.

import numpy as np
from circuits import circuit_builder
from qiskit import transpile
from qiskit_aer import AerSimulator
from tqdm import tqdm

def parameters_builder(num_parameter: int, alpha = np.pi/2):
    """ Compute the array of parametric phases, determining the 
        interferometric response function. Reverse the array order, having
        the last phase be the first to be injected in the circuit. """
    pars = [alpha/(2*idx - 1) for idx in range(1, num_parameter + 1)]
    return np.flip(pars)

# Qiskit simulator
simulator = AerSimulator()

# Hyperparameters 
numPhases = 300 # Number of unknown phase sampled
maxPhi = 2*np.pi # Default to [0,2pi]
numShots = 1000 # Number of shots for each phase

numParameters = np.array([1, 2, 4, 16]) # Number of phase parameters [D]
np.save('./output/data/numParameters.npy', numParameters) # Save

ampValues = np.array([0, 1/6, 1/2, 1])*np.pi # Amplitude coefficient [alpha]
np.save('./output/data/ampValues.npy', ampValues) # Save

argValues = np.array([1, 2, 4, 8]) # Response period multiplier [k]
np.save('./output/data/argValues.npy', argValues) # Save

# Input sampling methods
inPhases = np.arange(numPhases)*maxPhi/numPhases # Equally spaced 
# phiRNG = np.random.default_rng(seed = 2023) # Uniformly distributed
# inPhases = np.array(sorted(phiRNG.uniform(0, maxPhi, numPhases))) 
np.save('./output/data/inPhases.npy', inPhases) # Save

# Simulation 1 (with respect to numParameters, ampValues fixed, argValues = 1)
outValuesPars = np.zeros((len(inPhases), numShots, len(numParameters)))
ampValueFX = np.pi/2 # ampValues
for parIdx in tqdm(range(len(numParameters))):
    phiParameters = parameters_builder(numParameters[parIdx], ampValueFX)
    
    for idx in range(len(inPhases)):
        qc = circuit_builder(inPhases[idx], phiParameters)
        qc = transpile(qc, simulator)
        result = simulator.run(qc, shots = numShots, memory = True).result()
        outValuesPars[idx, :, parIdx] = result.get_memory(qc) 
        
np.save('./output/data/outValuesPars.npy', outValuesPars) # Save
        
# Simulation 2 (with respect to ampValues, numParameters fixed, argValues = 1)
outValuesAmps = np.zeros((len(inPhases), numShots, len(ampValues)))
numParameterFX = 16 # numParameter
for ampIdx in tqdm(range(len(ampValues))):
    phiParameters = parameters_builder(numParameterFX, ampValues[ampIdx])
    
    for idx in range(len(inPhases)):
        qc = circuit_builder(inPhases[idx], phiParameters)
        qc = transpile(qc, simulator)
        result = simulator.run(qc, shots = numShots, memory = True).result()
        outValuesAmps[idx, :, ampIdx] = result.get_memory(qc) 
        
np.save('./output/data/outValuesAmps.npy', outValuesAmps) # Save

# Simulation 3 (with respect to argValues, ampValues and numParameters fixed)
outValuesArgs = np.zeros((len(inPhases), numShots, len(argValues)))
numParameterFX = 16 # numParameters
ampValueFX = np.pi/2 # ampValues
for argIdx in tqdm(range(len(argValues))):
    phiParameters = parameters_builder(numParameterFX, ampValueFX)
    
    for idx in range(len(inPhases)):
        qc = circuit_builder(argValues[argIdx]*inPhases[idx], phiParameters)
        qc = transpile(qc, simulator)
        result = simulator.run(qc, shots = numShots, memory = True).result()
        outValuesArgs[idx, :, argIdx] = result.get_memory(qc) 
        
np.save('./output/data/outValuesArgs.npy', outValuesArgs) # Save
