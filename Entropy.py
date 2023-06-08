import os
import numpy as np


def entropy(file, z=0, y=0, x=0, dtype=0, n_clusters=None):
    if n_clusters:
        e, s = 0, 0
        for c in range(n_clusters):
            _rwa_output = file[:-4] + '_CL_' + str(c) + '.npy'
            im = np.load(_rwa_output, allow_pickle=True)
            
            e += _entropy(im) * im.shape[0]
            
            s += im.shape[0]
        e /= s
        
        return e
        
    else:
        G = np.fromfile(file, sep="", dtype=dtype)
        im = np.reshape(G, (x*y, z), order="F")
        
        e = _entropy(im)
        
        return e


def _entropy(data, bins=256):
    data = data.astype('int64')
    data[data < 0] += 65536
    data_normalized = data / np.max(np.abs(data))
    data_normalized = np.round(data_normalized, 4)

    bins = np.round(np.linspace(0, 1, 256) - 1/512, 4)
    bins[-1] = np.Inf

    cx = np.histogram(data_normalized, bins=bins)[0]
    normalized = cx / np.sum(cx)
    
    normalized = normalized[np.nonzero(normalized)]
    h = -sum(normalized * np.log2(normalized))
    
    return h