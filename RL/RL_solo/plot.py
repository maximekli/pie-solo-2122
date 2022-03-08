import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('training/training_results/solo-v1/solo-v1_s0/progress.txt',sep='\s+',header=0)  # verify path to progress.txt file
data = pd.DataFrame(data)

# Plotting the training cumulative reward data of each epoch
plt.plot(data['Epoch'], data['AverageEpRet'],'r-',label='Average')
plt.plot(data['Epoch'], data['MaxEpRet'],'r-*',label='Max')
plt.plot(data['Epoch'], data['MinEpRet'],'r--',label='Min')
plt.xlabel('Epoch')
plt.ylabel('EpRet')
plt.legend()
plt.show()

# Plotting the test cumulative reward data of each epoch
plt.plot(data['Epoch'], data['AverageTestEpRet'],'b-',label='Average')
plt.plot(data['Epoch'], data['MaxTestEpRet'],'b-*',label='Max')
plt.plot(data['Epoch'], data['MinTestEpRet'],'b--',label='Min')
plt.xlabel('Epoch')
plt.ylabel('TestEpRet')
plt.legend()
plt.show()

#plt.savefig('rewards.png')