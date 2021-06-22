"""
Example of CV procedure for GaussianProcessRegressor.
"""
from collections import Counter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold

# set matplotlib settings
mpl.rc('font', **{'family': 'serif', 'serif': ['Helvetica'], 'size': 7})
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.use('Agg')

# import color palette from seaborn
PALETTE = sns.color_palette('colorblind', n_colors=12)

# load dos data
DATA = np.loadtxt('../data/processed/dos_data.csv', delimiter=',', skiprows=1,
                  dtype='str')
E_C = DATA[:, 1].astype('float')
E_O = DATA[:, 2].astype('float')
E_N = DATA[:, 3].astype('float')
E_H = DATA[:, 4].astype('float')
ADS_DATA = [E_C, E_O, E_N, E_H]
LABELS = ['C', 'O', 'N', 'H']
DOS_DATA = DATA[:, 5:].astype('float')
DOS_ENERGY = np.linspace(-10, 10, 300)

# white kernel GP parameters to optimize over during GridSearchCV
PARAMETERS = {'alpha': [1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7,
                        1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
                        1e-2, 3e-2, 1e-1, 3e-1]}


for IND, E_ADS in enumerate(ADS_DATA):

    #  CV for full DOS
    print('=================')
    print('GP Full DOS ' + LABELS[IND] + ' 10-fold CV')
    print('=================')
    CV_ERROR = []
    GP_PARAMS = []
    CV_STD = []
    for _ in range(3):
        MEAN = 0
        KF = KF = KFold(10, shuffle=True)
        for train_index, test_index in KF.split(DOS_DATA):
            X_TRAIN, X_TEST = (DOS_DATA[train_index, :],
                               DOS_DATA[test_index, :])
            Y_TRAIN, Y_TEST = (E_ADS[train_index],
                               E_ADS[test_index])
            GP = GaussianProcessRegressor()
            CV = GridSearchCV(GP, PARAMETERS, cv=10, iid=False)
            CV.fit(X_TRAIN, Y_TRAIN)
            GP_PARAMS.append(CV.best_params_['alpha'])
            GP = CV.best_estimator_
            Y_PRED = GP.predict(X_TEST)
            MEAN += (mean_squared_error(Y_PRED, Y_TEST) ** .5) / 10
            CV_STD.append(mean_squared_error(Y_PRED, Y_TEST) ** .5)
        CV_ERROR.append(MEAN)
    CV_STD = np.asarray(CV_STD)
    CV_ERROR = np.asarray(CV_ERROR)
    print('CV STD: {}'.format(np.std(CV_STD)))
    print('CV error: {}'.format(np.sum(CV_ERROR) / CV_ERROR.size))
    print('Best GP alpha: {}'.format(Counter(GP_PARAMS).most_common(1)[0][0]) +
          ' count: {}'.format(Counter(GP_PARAMS).most_common(1)[0][1]))

    FIG, AX = plt.subplots(figsize=(3, 2))
    # CV for TruncatedSVD
    print('=================')
    print('GP TruncatedSVD ' + LABELS[IND] + ' 10-fold CV')
    print('=================')
    ERROR = []
    WRITE_DATA = pd.DataFrame()
    for n_components in range(1, 21):
        PC_T = TruncatedSVD(n_components=n_components)
        PCA_DATA = PC_T.fit_transform(DOS_DATA)
        CV_ERROR = []
        GP_PARAMS = []
        for _ in range(3):
            MEAN = 0
            KF = KFold(10, shuffle=True)
            for train_index, test_index in KF.split(DOS_DATA):
                X_TRAIN, X_TEST = (PCA_DATA[train_index, :],
                                   PCA_DATA[test_index, :])
                Y_TRAIN, Y_TEST = (E_ADS[train_index],
                                   E_ADS[test_index])
                GP = GaussianProcessRegressor()
                CV = GridSearchCV(GP, PARAMETERS, cv=10, iid=False)
                CV.fit(X_TRAIN, Y_TRAIN)
                GP_PARAMS.append(CV.best_params_['alpha'])
                GP = CV.best_estimator_
                Y_PRED = GP.predict(X_TEST)
                MEAN += (mean_squared_error(Y_PRED, Y_TEST) ** .5) / 10
            CV_ERROR.append(MEAN)
        CV_ERROR = np.asarray(CV_ERROR)
        ERROR.append(np.sum(CV_ERROR) / CV_ERROR.size)
        print('Number of components: {}'.format(n_components))
        print('CV error: {}'.format(np.sum(CV_ERROR) / CV_ERROR.size))
        print('Best GP alpha: {}'.format(Counter(GP_PARAMS).
                                         most_common(1)[0][0]) +
              ' count: {}'.format(Counter(GP_PARAMS).most_common(1)[0][1]))
    plt.plot(range(1, 21), ERROR, color=PALETTE[0])
    plt.legend(frameon=False)
    AX.set_xlabel(r'Number of singular values considered')
    AX.set_ylabel(r'$\mathregular{RMSE_{CV}}$ (eV)')
    plt.show()

    FIG, AX = plt.subplots(figsize=(3, 2))
    # CV for standard PCA
    print('=================')
    print('GP PCA ' + LABELS[IND] + ' 10-fold CV')
    print('=================')
    ERROR = []
    STD = []
    WRITE_DATA = pd.DataFrame()
    for n_components in range(1, 21):
        PC_T = PCA(n_components=n_components)
        PCA_DATA = PC_T.fit_transform(DOS_DATA)
        CV_ERROR = []
        CV_STD = []
        GP_PARAMS = []
        for _ in range(3):
            MEAN = 0
            KF = KFold(10, shuffle=True)
            for train_index, test_index in KF.split(DOS_DATA):
                X_TRAIN, X_TEST = (PCA_DATA[train_index, :],
                                   PCA_DATA[test_index, :])
                Y_TRAIN, Y_TEST = (E_ADS[train_index],
                                   E_ADS[test_index])
                GP = GaussianProcessRegressor()
                CV = GridSearchCV(GP, PARAMETERS, cv=10, iid=False)
                CV.fit(X_TRAIN, Y_TRAIN)
                GP_PARAMS.append(CV.best_params_['alpha'])
                GP = CV.best_estimator_
                Y_PRED = GP.predict(X_TEST)
                MEAN += (mean_squared_error(Y_PRED, Y_TEST) ** .5) / 10
                CV_STD.append(mean_squared_error(Y_PRED, Y_TEST) ** .5)
            CV_ERROR.append(MEAN)
        CV_ERROR = np.asarray(CV_ERROR)
        CV_STD = np.asarray(CV_STD)
        STD.append(np.std(CV_STD))
        ERROR.append(np.sum(CV_ERROR) / CV_ERROR.size)
        print('Number of components: {}'.format(n_components))
        print('CV error: {}'.format(ERROR[n_components-1]))
        print('CV STD: {}'.format(STD[n_components-1]))
        print('Best GP alpha: {}'.format(Counter(GP_PARAMS).
                                         most_common(1)[0][0]) +
              ' count: {}'.format(Counter(GP_PARAMS).most_common(1)[0][1]))
    plt.plot(range(1, 21), ERROR, color=PALETTE[0])
    plt.legend(frameon=False)
    AX.set_xlabel(r'Number of singular values considered')
    AX.set_ylabel(r'$\mathregular{RMSE_{CV}}$ (eV)')
    plt.show()

    # CV for kernel PCA
    FIG, AX = plt.subplots(figsize=(3, 2))
    GAMMAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    print('=================')
    print('GP Kernel PCA ' + LABELS[IND] + ' 10-fold CV')
    print('=================')
    for j, gamma in enumerate(GAMMAS):
        ERROR = []
        for n_components in range(1, 21):
            print(n_components)
            PC_T = KernelPCA(n_components=n_components, kernel='rbf',
                             gamma=gamma)
            PCA_DATA = PC_T.fit_transform(DOS_DATA)
            CV_ERROR = []
            GP_PARAMS = []
            for _ in range(3):
                MEAN = 0
                KF = KFold(10, shuffle=True)
                for train_index, test_index in KF.split(DOS_DATA):

                    for i in range(1):
                        X_TRAIN, X_TEST = (PCA_DATA[train_index, :],
                                           PCA_DATA[test_index, :])
                        Y_TRAIN, Y_TEST = (E_ADS[train_index],
                                           E_ADS[test_index])
                        GP = GaussianProcessRegressor()
                        CV = GridSearchCV(GP, PARAMETERS, cv=10, iid=False)
                        CV.fit(X_TRAIN, Y_TRAIN)
                        GP_PARAMS.append(CV.best_params_['alpha'])
                        GP = CV.best_estimator_
                        Y_PRED = GP.predict(X_TEST)
                        MEAN += (mean_squared_error(Y_PRED, Y_TEST) ** .5) / 10
                CV_ERROR.append(MEAN)
            CV_ERROR = np.asarray(CV_ERROR)
            ERROR.append(np.sum(CV_ERROR) / CV_ERROR.size)
            print('n_components: {}'.format(n_components))
            print('gamma" {}'.format(gamma))
            print('CV error: {}'.format(np.sum(CV_ERROR) / CV_ERROR.size))
            print('Best GP alpha: {}'.format(Counter(GP_PARAMS).
                                             most_common(1)[0][0]) +
                  ' count: {}'.format(Counter(GP_PARAMS).most_common(1)[0][1]))

            plt.plot(range(1, 21), ERROR,
                     label=r'$\mathregular{\gamma}$'+'={}'.format(gamma),
                     color=PALETTE[j])
        plt.legend(frameon=False)
        AX.set_xlabel(r'Number of principal components used')
        AX.set_ylabel(r'$\mathregular{RMSE_{CV}}$ (eV)')
        plt.show()
