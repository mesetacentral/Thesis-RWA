import numpy as np
from sklearn.cluster import KMeans

from RWA import *
from RWANN import *
from Entropy import entropy


def RWAC_Transform(raw_image, z, y, x, dtype, output, n_clusters=4, R=False, compression_technology='linear_regression', layer=-1, train_split=0.01, verbose=False):
    G = np.fromfile(raw_image, sep="", dtype=dtype)
    image = np.reshape(G, (x*y, z), order="F").astype('int32')
    l = int(np.ceil(np.log2(z)))
    
    # KMeans clustering
    km = KMeans(n_clusters)
    km_clusters = km.fit_predict(image)
    np.save(output[:-4] + '_KM.npy', km_clusters)
    
    # compression
    for c in range(n_clusters):
        if verbose:
            print('Cluster ' + str(c))
        
        _sifile = output[:-4] + '_CL_' + str(c) + '_SI.npy'
        _output = output[:-4] + '_CL_' + str(c) + '.npy'
        
        im = image[km_clusters == c].astype('int32')

        RWAim = RWANN(im, l, _sifile, R, compression_technology, layer, train_split, verbose)
        RWAim = RWAim.astype('int32')
        np.save(_output, RWAim)
    

def inv_RWAC_Transform(raw_image, z, y, x, dtype, output, R=False, compression_technology='linear_regression', layer=-1):
    km_clusters = np.load(raw_image[:-4] + '_KM.npy')
    n_clusters = np.unique(km_clusters).shape[0]
    recovered_image = np.empty(shape=(km_clusters.shape[0], z), dtype='int32')
    l = int(np.ceil(np.log2(z)))

    for c in range(n_clusters):
        _raw_image = raw_image[:-4] + '_CL_' + str(c) + '.npy'
        _sifile = raw_image[:-4] + '_CL_' + str(c) + '_SI.npy'
        _output = output[:-4] + '_CL_' + str(c) + '.npy'
        
        RWAim = np.load(_raw_image)
        im = inv_RWANN(RWAim, l, _sifile, R, compression_technology, layer)
        np.save(_output, im)

        recovered_image[km_clusters == c] = im

    np.save(output, recovered_image)
