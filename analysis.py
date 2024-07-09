### This script processes and plots the interferometric data 
### at the output of simulation.py

import numpy as np
import matplotlib.pyplot as plt

def step(xArr, minVal:float, maxVal:float, period = 2*np.pi):
    """ Compute the ideal step function. """
    yArr = np.zeros(xArr.shape)
    for idx, val in np.ndenumerate(xArr):
        if (val % period) <= (period / 2):
            yArr[idx] = maxVal
        else:
            yArr[idx] = minVal
    return yArr

# Load the data
x = np.load('./output/data/inPhases.npy')
argValues = np.load('./output/data/argValues.npy')
ampValues = np.load('./output/data/ampValues.npy')
numParameters = np.load('./output/data/numParameters.npy')

# Shape (numPhases, numShots, len(numParameters))
dataPars = np.load('./output/data/outValuesPars.npy')

# Shape (numPhases, numShots, len(ampValues))
dataAmps = np.load('./output/data/outValuesAmps.npy')

# Shape (numPhases, numShots, len(argValues))
dataArgs = np.load('./output/data/outValuesArgs.npy')

# Analysis 1 (with respect to Simulation 1)
yPars = np.zeros((dataPars.shape[0], dataPars.shape[2]))
for parIdx in range(dataPars.shape[2]):
    for idx in range(dataPars.shape[0]):
        unique, count = np.unique(dataPars[idx,:, parIdx], return_counts=True)
        counts = dict(zip(unique, count)) # Counts
        tot = sum(counts.values(), 0.0)
        freq = {key: val/tot for key, val in counts.items()} # Frequencies
        yPars[idx, parIdx] = freq.get(0, 0) # Probability of getting |0>  
 
# Analysis 2 (with respect to Simulation 2)  
yAmps = np.zeros((dataAmps.shape[0], dataAmps.shape[2]))
for ampIdx in range(dataAmps.shape[2]):
    for idx in range(dataAmps.shape[0]):
        unique, count = np.unique(dataAmps[idx,:, ampIdx], return_counts=True)
        counts = dict(zip(unique, count)) # Counts
        tot = sum(counts.values(), 0.0)
        freq = {key: val/tot for key, val in counts.items()} # Frequencies
        yAmps[idx, ampIdx] = freq.get(0, 0) # Probability of getting |0>  

# Analysis 3 (with respect to Simulation 3) 
yArgs = np.zeros((dataPars.shape[0], dataPars.shape[2]))        
for argIdx in range(dataArgs.shape[2]):
    for idx in range(dataArgs.shape[0]):
        unique, count = np.unique(dataArgs[idx,:, argIdx], return_counts=True)
        counts = dict(zip(unique, count)) # Counts
        tot = sum(counts.values(), 0.0)
        freq = {key: val/tot for key, val in counts.items()} # Frequencies
        yArgs[idx, argIdx] = freq.get(0, 0) # Probability of getting |0>     

# Canvas
plt.rcParams.update({'font.size': 14})
markers = ['o', 's', '^', '>', 'v','<']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',\
          'tab:purple','tab:brown']
mSize = 1.5 # Markersize
lWidth = 0.0 # Linewidth
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi = 400,\
                                 constrained_layout = True)

# Plot 1 (with respect to Simulation 1)
for idx in range(dataPars.shape[2]):
    ax1.plot(x, yPars[:,idx], linestyle = 'None',\
                 color = colors[idx], marker = markers[idx],\
                     markersize = mSize, linewidth = lWidth,\
                         label = f"""D = {numParameters[idx]}""")
ax1.plot(x, step(x, 0, 1), '--', markersize = mSize,\
         color = 'black', linewidth = 1) 

ax1.set_xlabel(r"""Input phase [rad]
(a)""")
ax1.set_ylabel(r'Probability')
ax1.legend()


# Plot 2 (with respect to Simulation 2)
labels = [r"$\alpha = 0$", r"$\alpha = \pi/6$",\
          r"$\alpha = \pi/2$", r"$\alpha =  \pi$"]
for idx in range(dataAmps.shape[2]):
    ax2.plot(x, yAmps[:,idx], linestyle = 'None',\
                 color = colors[idx], marker = markers[idx],\
                     markersize = mSize, linewidth = lWidth,\
                         label = labels[idx])   
ax2.plot(x, step(x, 0, 1), '--', markersize = mSize,\
         color = 'black', linewidth = 1)

ax2.set_xlabel(r"""Input phase [rad]
(b)""")
ax2.set_ylabel(r'Probability')
ax2.legend()

# Plot 3 (with respect to Simulation 3)
labels = [r"$k_j = 1$", r"$k_j = 2$",\
          r"$k_j = 4$", r"$k_j = 8$"]
for idx in range(dataArgs.shape[2]): 
    ax3.plot(x, yArgs[:,idx]/argValues[idx], linestyle = 'None',\
                 color = colors[idx], marker = markers[idx],\
                     markersize = mSize, linewidth = lWidth,\
                         label = labels[idx])  
    if idx == idx:
        ax3.plot(x[:len(x)//(2**idx)], step(x[:len(x)//(2**idx)],\
                            0, 1/argValues[idx],\
                             period = 2*np.pi/argValues[idx]), '--',\
                                 markersize = mSize, color = 'black',\
                                     linewidth = 1) 

ax3.set_xlabel(r"""Input phase [rad]
(c)""")
ax3.set_ylabel(r'Probability')
ax3.legend()

plt.savefig('output/plots/Simulation.pdf', format='pdf') # Save
plt.show()
