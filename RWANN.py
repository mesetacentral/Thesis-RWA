from sklearn.linear_model import *
from sklearn.preprocessing import *
import torch
from torch.nn import *
from torch.utils import *
import numpy as np
from joblib import dump, load
import time
from NeuralNetwork import *


def RWANN_Transform(raw_image, z, y, x, dtype, output, R=False, compression_technology='linear_regression', scale='all', train_split=0.01, verbose=False):
    G = np.fromfile(raw_image, sep="", dtype=dtype)
    im = np.reshape(G, (x*y, z), order="F").astype('int32')
    l = int(np.ceil(np.log2(z)))
    
    sifile = output[:-4] + '_SI.npy'

    RWAim = RWANN(im, l, sifile, R, compression_technology, scale, train_split, verbose)
    RWAim = RWAim.astype('int32')
    np.save(output, RWAim)

    print('\n Image: {} \n size: ({}, {}, {}) \n Transformed: {} \n'.format(raw_image, z, y, x, output));
    

def RWANN(im, l=1, sifile=None, R=False, compression_technology='linear_regression', scale=None, train_split=0.01, verbose=False):
    t1 = time.time()
    y, z = im.shape
    L, H = [], []
    fijo = None
    data = im.copy()

    for i in range(0, l):
        L, H = RWANN1l(data, sifile, R, compression_technology, scale, train_split, verbose)
        
        try:
            fijo = np.hstack((H, fijo))
        except ValueError:
            fijo = H.copy()
        data = L.copy()
        
    pim = np.hstack((L, fijo))
    
    return pim


def RWANN1l(im, sifile=None, R=False, compression_technology='linear_regression', scale=None, train_split=0.01, verbose=False):        
    y, z = im.shape
    
    p = int(np.round(z / 2))
    q = int(np.floor(z / 2))
    
    if p % 2 == 0 and abs(p - (z/2)) > 0.4 and p == q:
        p += 1
    
    H = np.zeros((y, q))
    L = np.zeros((y, p))
    
    for j in range(0, q):
        H[:, j] = im[:, 2*j] - im[:, 2*j+1] # details
        L[:, j] = im[:, 2*j+1] + np.floor(H[:, j] / 2) # approximation
    
    if z % 2 != 0:
        L[:, -1] = im[:, -1]


    if compression_technology == 'linear_regression' or L.shape[1] != scale:
        M = fit_NNregression(L, H, sifile, R, compression_technology='linear_regression', train_split=train_split, verbose=verbose)  
    else:
        M = fit_NNregression(L, H, sifile, R, compression_technology=compression_technology, train_split=train_split, verbose=verbose)
    
    H = H - np.round(M)
    
    return L, H


def fit_NNregression(X, y, sifile=None, R=False, compression_technology='linear_regression', train_split=0.01, verbose=False):
    # np.save(sifile[:-7] + '_XT_' + str(X.shape[1]) + '.npy', X)
    # np.save(sifile[:-7] + '_YT_' + str(X.shape[1]) + '.npy', y)

    X_shape = X.shape[1]

    if R:
        X = np.reshape(X[:, 0], (X.shape[0], 1))

        poly = PolynomialFeatures(3)
        X = poly.fit_transform(X)


    if compression_technology == 'linear_regression':
        if not R:
            poly = PolynomialFeatures(1)
            X = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X, y)

        M = model.predict(X)
        
        dump(model, 'output//' + compression_technology + '_' + str(X_shape) + '.npy')
        
    elif compression_technology == 'local_NNRegressor':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        dump(scaler, sifile[:-4] + '_' + str(X_shape) + 'SC.npy')
        
        n_indices = int(X.shape[0] * train_split)
        indices = np.random.randint(0, X.shape[0], n_indices)

        X = torch.Tensor(X)
        y = torch.Tensor(y)

        # TODO: pass model, criterion, optimizer to this function
        model = local_NNRegressor(X.shape[1], 16, y.shape[1])
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.2, weight_decay=0.9)

        model.fit(
            X[indices, :], 
            y[indices, :], 
            criterion, 
            optimizer, 
            epochs = 50,
            batch_size = 32
        )

        torch.save(model, sifile[:-4] + '_' + str(X_shape) + '.npy')

        M = model(X)
        M = M.detach().numpy()

    elif compression_technology.startswith('NNRegressor'):
        model = load('output//' + compression_technology + '_' + str(X_shape) + '.npy')
        scaler = load('output//Scaler_' + str(X_shape) + '.npy')
        
        X = scaler.transform(X)
        
        M = model(X)
        M = M.detach().numpy()
        
    else:
        model = load('output//' + compression_technology + '_' + str(X_shape) + '.npy')
        scaler = load('output//Scaler_' + str(X_shape) + '.npy')
    
        X = scaler.transform(X)
        
        M = model.predict(X)
        
    if verbose:
        print('Compressed scale ' + str(X_shape) + ' using ' + compression_technology)

    M = np.reshape(M, y.shape)

    return M



def inv_RWANN_Transform(raw_image, z, y, x, dtype, output, R, compression_technology='linear_regression', scale=-1):
    RWAim = np.load(raw_image, allow_pickle=True)
    l = int(np.ceil(np.log2(z)))
    
    sifile = raw_image[:-4] + '_SI.npy'
    
    im = inv_RWANN(RWAim, l, sifile, R, compression_technology, scale)
    np.save(output, im)
    
    print('\n Transformed: {} \n size: ({}, {}, {}) \n Recovered: {} \n'.format(raw_image, z, y, x, output));


def inv_RWANN(im, l=1, sifile=None, R=False, compression_technology='linear_regression', scale=-1):
    y, z = im.shape
    
    data = im
    
    P = []
    Q = []
    
    for i in range(0, l):
        p = int(np.ceil(z/2))
        q = int(np.floor(z/2))
        P.append(p)
        Q.append(q)
        
        z = p
    
    for i in reversed(range(0, l)):
        p = P[i]
        q = Q[i]
        
        L = data[:, :p]
        H = data[:, p:p+q]
                
        aux = inv_RWANN1l(L, H, sifile, R, compression_technology, scale)
        data[:, 0:p+q] = aux

    im=data
    
    return im
    
    
def inv_RWANN1l(L, H, sifile=None, R=False, compression_technology='linear_regression', scale=-1):
    if compression_technology == 'linear_regression' or L.shape[1] != scale:
        M = generate_NNregression(L, sifile, R, compression_technology='linear_regression')
    else:
        M = generate_NNregression(L, sifile, R, compression_technology=compression_technology)    
    
    H = H + np.round(M)

    q = H.shape[1]
    p = L.shape[1]
    z = p+q

    im = np.zeros((L.shape[0], z))
    
    for j in range(0, q):
        im[:, 2*j+1] = L[:, j] - np.floor(H[:, j] / 2)
        
        im[:, 2*j] = im[:, 2*j+1] + H[:, j]
        
    if z % 2 != 0:
        im[:, 2*q] = L[:, -1]
        
    return im


def generate_NNregression(X, sifile=None, R=False, compression_technology='linear_regression'):

    X_shape = X.shape[1]

    if R:
        X = np.reshape(X[:, 0], (X.shape[0], 1))

        poly = PolynomialFeatures(3)
        X = poly.fit_transform(X)


    if compression_technology == 'linear_regression':
        if not R:
            poly = PolynomialFeatures(1)
            X = poly.fit_transform(X)
        
        model = load('output//' + compression_technology + '_' + str(X_shape) + '.npy')

        M = model.predict(X)
        
    elif compression_technology == 'local_NNRegressor':
        model = torch.load(sifile[:-4] + '_' + str(X_shape) + '.npy')
        model.eval()
        scaler = load(sifile[:-4] + '_' + str(X_shape) + 'SC.npy')
        
        X = scaler.transform(X)

        X = torch.Tensor(X)
        M = model(X)
        M = M.detach().numpy()
    
    elif compression_technology.startswith('NNRegressor'):
        model = load('output//' + compression_technology + '_' + str(X_shape) + '.npy')
        scaler = load('output//Scaler_' + str(X_shape) + '.npy')
            
        X = scaler.transform(X)
            
        M = model(X)
        M = M.detach().numpy()

    else:
        model = load('output//' + compression_technology + '_' + str(X_shape) + '.npy')
        scaler = load('output//Scaler_' + str(X_shape) + '.npy')
        
        X = scaler.transform(X)
        M = model.predict(X)


    if len(M.shape) == 1:
        M = np.reshape(M, (M.shape[0], 1))
    
    return M    
            