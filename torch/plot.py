import numpy as np
import matplotlib.pyplot as plt
# filename = 'ising_trg_square.txt'
# filename = 'ising_ctmrg_honeycomb.txt'
filename = 'ising_ctmrg_square.txt'
data = np.loadtxt(filename)
beta = data[:,0]
T = 1/beta
energy_den = data[:,2]
specific_heat = data[:,3]
mag = abs(data[:,4])
plt.figure(figsize=(12,6))
plt.subplot(1, 3, 1)
# plt.title(filename)
plt.plot(T, energy_den, marker='o',linestyle='-.')
plt.ylabel('energy density')
plt.grid()
plt.subplot(1, 3, 2)
# marker = ['o','+','s','v','x']
# lstyle = [':','-', '-.', '--', '-']
# label = ['1','2','3','4','5']
plt.plot(T, specific_heat, marker='o',linestyle='-.')
# plt.xlabel('$\\beta$')
plt.xlabel('T')
# print(specific_heat)
plt.ylabel('specific heat')
# plt.legend()
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(T, mag, marker='o',linestyle='-.')
plt.ylabel('M')

plt.show()
#print(data)
