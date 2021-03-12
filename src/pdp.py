"""
Example of PDP for GaussianProcessRegressor.
"""
from interpret.glassbox import ExplainableBoostingRegressor
from interpret.blackbox import PartialDependence
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV


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

E_C = DATA[:, 1].astype('float')
E_O = DATA[:, 2].astype('float')
E_N = DATA[:, 3].astype('float')
E_H = DATA[:, 4].astype('float')
ADS_DATA = [E_C, E_O, E_N, E_H]
DOS_DATA = DATA[:, 5:].astype('float')
PCA_O = PCA(n_components=10, random_state=0)
PCA_DATA = PCA_O.fit_transform(DOS_DATA)

PARAMETERS = {'alpha': [1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7,
                        1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
                        1e-2, 3e-2, 1e-1, 3e-1]}
for i in range(2):
    FIG, AX = plt.subplots(figsize=(3.25, 2))
    for IND, E_ADS in enumerate(ADS_DATA):

        GP = GaussianProcessRegressor()
        CV = GridSearchCV(GP, PARAMETERS, cv=10, iid=False)
        CV.fit(PCA_DATA, E_ADS)
        GP = CV.best_estimator_
        PDP = PartialDependence(predict_fn=GP.predict, data=PCA_DATA)
        PDP = PDP.explain_global(name='Partial Dependence')
        data_dict = PDP.data(i)
        AX.plot(data_dict['names'], data_dict['scores'])
        y_hi = data_dict['upper_bounds']
        y_lo = data_dict['lower_bounds']
        AX.fill_between(data_dict['names'], y_hi, y_lo, alpha=0.1)
        AX.set_ylabel('Adsorption energy (eV)')
        if i == 0:
            AX.set_xlabel('First principal component')
        if i == 1:
            AX.set_xlabel('Second principal component')
    plt.show()
