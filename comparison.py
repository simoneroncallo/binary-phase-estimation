### This script compares the theoretical response function (without sampling), 
### with the truncated Fourier series of the square wave

import numpy as np
import matplotlib.pyplot as plt

def fourier_builder(inputs, numMode:int, alpha:float, numPeriods:float):
    """ Return a single Fourier mode, for a given input array and 
        parameters alpha and numPeriods. Enhance superimposed visualization 
        by downscaling the output to the corresponding value of numPeriods. """
    outputs = np.zeros(inputs.shape)
    for idx, xVal in np.ndenumerate(inputs):
        b = 2*np.sin(alpha)/(np.pi*(2*numMode + 1))
        outputs[idx] = b*np.sin((2*numMode + 1)*numPeriods*xVal)
    return outputs

def step(xArr, alpha:float, period = 2*np.pi):
    """ Compute the expected square wave. """
    yArr = np.zeros(x.shape)
    for idx, val in np.ndenumerate(xArr):
        if (val % period) <= (period / 2):
            yArr[idx] = 1
        else:
            yArr[idx] = 0
    return yArr

def phaseShift(phase):
    """ Phase shift transformation. """
    return np.array([[1, 0], [0, np.exp(1j*phase)]])

def module(var, par):
    """ Interferometric layer. """
    Had = np.array([[1, 1], [1, -1]])/np.sqrt(2) # Hadamard gate
    return phaseShift(var) @ ( Had @ ( phaseShift(par) @ Had ) ) 

def unitary(var, pars): 
    """ Compute the total unitary by composing multiple interferometric
    layers. """
    output = np.identity(2)
    for par in pars:
        output = output @ module(var, par) @ module(var, 0)
    return output

def response_builder(inputs, pars, k):
    """ Compute the probability of getting 0 in the x-basis, for a given
    input array. """
    outputs = np.zeros(inputs.shape)
    Had = np.array([[1, 1], [1, -1]])/np.sqrt(2) # Hadamard gate
    state = np.array([[1], [0]])
    for idx, xVal in np.ndenumerate(inputs):
        amplitudes = ( Had @ ( unitary(k*xVal, pars) @ state ) )
        outputs[idx] = np.absolute(amplitudes[0,0])**2
    return outputs
        
x = np.load('./output/data/inPhases.npy')
numModes = np.load('./output/data/numParameters.npy')
numModes = np.array([1, 2, 3, 4, 5]) 

# Hyperparameters 
alpha = np.pi/2 # Amplitude
k = 1 # Period compression

# Fourier modes
maxDepth = numModes[-1]
normalModes = np.zeros((x.shape[0], maxDepth))
for modeIdx in range(maxDepth):
    normalModes[:, modeIdx] = fourier_builder(x, modeIdx, alpha, k)

# Compute the truncated Fourier series
partialSeries = np.zeros((x.shape[0], len(numModes)))
for truncIdx in range(len(numModes)):
    for modeIdx in range(0, numModes[truncIdx]): 
        partialSeries[:, truncIdx] = partialSeries[:, truncIdx] +\
            normalModes[:, modeIdx] 
partialSeries = partialSeries + 0.5 # Add a0 coefficient
for truncIdx in range(len(numModes)): # Regularization to [0,1]
    shift = np.min(partialSeries[:, truncIdx]) # Minimum
    norm = np.max(partialSeries[:, truncIdx] + np.abs(shift))
    partialSeries[:, truncIdx] =\
        partialSeries[:, truncIdx] + np.abs(shift) # Minimum shifted to 0
    partialSeries[:, truncIdx] =\
        partialSeries[:, truncIdx]/norm # Maximum normalized to 1

# Response 1 [beta = 1]    
responseFunction = np.zeros((x.shape[0], len(numModes)))
for truncIdx in range(len(numModes)):
    parPhases = np.array([alpha/(idx)\
                              for idx in range(1, numModes[truncIdx] + 1)]) 
    responseFunction[:, truncIdx] = response_builder(x, parPhases, k)

# Response 2 [beta = 2]  
responseFunctionOdd = np.zeros((x.shape[0], len(numModes)))
for truncIdx in range(len(numModes)):
    parPhases = np.array([alpha/(2*idx - 1)\
                              for idx in range(1, numModes[truncIdx] + 1)]) 
    responseFunctionOdd[:, truncIdx] = response_builder(x, parPhases, k)

# Canvas
plt.rcParams.update({'font.size': 14})
markers = ['o', 's', '^', '>', 'v','<']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',\
          'tab:purple','tab:brown']
mSize = 1.5 # Markersize
lWidth = 0.0 # Linewidth

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi = 400,\
                          constrained_layout = True)

# Plot 1
for idx in range(numModes.shape[0]):
    ax1.plot(x, partialSeries[:,idx], 'None',\
                 color = colors[idx], marker = markers[idx],\
                     markersize = mSize, linewidth = lWidth,\
                         label = f"""D = {numModes[idx]}""")

ax1.plot(x, step(x, alpha, period = 2*np.pi/k), '--', markersize = mSize,\
         color = 'black', linewidth = 1)    
ax1.set_xlabel(r"""Input phase [rad]
(a)""")
ax1.set_ylabel(r'Probability')
ax1.legend()

# Plot 2
for idx in range(numModes.shape[0]):
    ax2.plot(x, responseFunction[:,idx], 'None',\
                 color = colors[idx], marker = markers[idx],\
                     markersize = mSize, linewidth = lWidth,\
                         label = f"""D = {numModes[idx]}""")

ax2.plot(x, step(x, alpha, period = 2*np.pi/k), '--', markersize = mSize,\
         color = 'black', linewidth = 1)
ax2.set_xlabel(r"""Input phase [rad]
(b)""")
ax2.set_ylabel(r'Probability')
ax2.legend()

# Plot 3
for idx in range(numModes.shape[0]):
    ax3.plot(x, responseFunctionOdd[:,idx], 'None',\
                 color = colors[idx], marker = markers[idx],\
                     markersize = mSize, linewidth = lWidth,\
                         label = f"""D = {numModes[idx]}""")

ax3.plot(x, step(x, alpha, period = 2*np.pi/k), '--', markersize = mSize,\
         color = 'black', linewidth = 1)
ax3.set_xlabel(r"""Input phase [rad]
(c)""")
ax3.set_ylabel(r'Probability')
ax3.legend()

plt.savefig('output/plots/Comparison.pdf', format='pdf')
plt.show()



