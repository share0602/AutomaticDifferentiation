import numpy as np
import matplotlib.pyplot as plt
filename = 'ising_trg_square.txt'
data = np.loadtxt(filename)
beta = data[:,0]

energy_den = data[:,2]
specific_heat = data[:,3]

plt.figure(figsize=(8,6))
plt.title(filename)
plt.subplot(1, 2, 1)
plt.plot(beta, energy_den, marker='o',linestyle='-.')
plt.xlabel('$\\beta$')
plt.ylabel('energy density')
plt.grid()
plt.subplot(1, 2, 2)
marker = ['o','+','s','v','x']
lstyle = [':','-', '-.', '--', '-']
label = ['1','2','3','4','5']
plt.plot(beta, specific_heat, marker='o',linestyle='-.')
plt.xlabel('$\\beta$')
plt.ylabel('specific heat')
# plt.legend()
plt.grid()
plt.show()
#print(data)
