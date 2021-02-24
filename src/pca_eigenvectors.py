"""
Example of DOS reconstructions for alloys.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# load data
DATA = np.loadtxt('../data/processed/dos_data.csv', delimiter=',', skiprows=1,
                  dtype='str')
DOS_ENERGY = np.loadtxt('../data/processed/dos_data.csv', delimiter=',',
                        dtype='str', skiprows=0)
DOS_ENERGY = DOS_ENERGY[0, 5:]
for i in range(DOS_ENERGY.size):
    DOS_ENERGY[i] = DOS_ENERGY[i].split()[2]
DOS_ENERGY = DOS_ENERGY.astype('float')
DOS_DATA = DATA[:, 5:].astype('float')

PCA_DOS = PCA(n_components=3, random_state=0)
X_TRAIN = PCA_DOS.fit_transform(DOS_DATA)

# first PC reconstructions
CHANGE = np.linspace(-0.5, 0.5, 5)
FIG, AX = plt.subplots()
AX.set_xlabel('Energy (eV)')
AX.set_ylabel('Density of states (a.u.)')
for j, i in enumerate(CHANGE):
    plt.plot(DOS_ENERGY, PCA_DOS.mean_+PCA_DOS.components_[0, :]*i,
             label='PC = {:.2f}'.format(i))
plt.legend()

# second PC reconstructions
FIG, AX = plt.subplots()
AX.set_xlabel('Energy (eV)')
AX.set_ylabel('Density of states (a.u.)')
CHANGE = np.linspace(-0.4, 0.4, 5)
for j, i in enumerate(CHANGE):
    plt.plot(DOS_ENERGY, PCA_DOS.mean_+PCA_DOS.components_[1, :]*i,
             label='PC = {:.2f}'.format(i))
plt.legend()
plt.show()
