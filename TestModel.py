import os
import numpy as np
import csv
import time

from RWAC import *
from RWANN import *
from NeuralNetwork import *

def test_model(image_name, n_clusters, R, compression_technology, scale, train_split, verbose=False, write_results=False):
    print('Image: ', image_name)
    
    z, x, y = image_name.split('_')[-6:-3]
    x = int(x)
    y = int(y)
    z = int(z.split('.')[-1])
    dtype = '<u2'

    """
    z, y, x = image_name[:-4].split('x')[-3:]
    x = int(x)
    y = int(y)
    z = int(z.split('-')[-1])
    dtype = '>i2'
    """

    l = int(np.ceil(np.log2(z)))
    
    output_folder = 'output'
    rwa_output = output_folder + '//RWA_' + image_name[:-4] + '.npy'
    true_output = 'output_true//RWA_' + image_name
    inv_output = output_folder + '//inv_RWA_' + image_name[:-4] + '.npy'
    
    t_rwa = time.time()
    RWAC_Transform(image_name, z, y, x, dtype, rwa_output, n_clusters, R, compression_technology, scale, train_split, verbose)
    t_rwa = time.time() - t_rwa
    t_inv = time.time()
    inv_RWAC_Transform(rwa_output, z, y, x, dtype, inv_output, R, compression_technology, scale)
    t_inv = time.time() - t_inv
    
    or_entropy = entropy(image_name, z=z, y=y, x=x, dtype=dtype)
    tr_entropy = entropy(rwa_output, n_clusters=n_clusters)
    
    print('OR ENTROPY: ', str(or_entropy))
    print('TR ENTROPY: ', str(tr_entropy))
    print('RWA TIME: ', str(t_rwa))
    print('INV TIME: ', str(t_inv))

    if write_results:
        with open('model_results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([image_name, z, y, x, dtype, rwa_output, n_clusters, R, compression_technology, scale, train_split, t_rwa, t_inv, t_rwa+t_inv, or_entropy, tr_entropy])

    """
    im = np.load(inv_output, allow_pickle=True)
    G = np.fromfile(image_name, sep="", dtype=dtype)
    true_im = np.reshape(G, (x*y, z), order="F")

    same_recovered = np.equal(im, true_im).all()
    
    if same_recovered:
        print('IDENTICAL')
        
        print('Original entropy ' + str(entropy(image_name, z=z, y=y, x=x, dtype=dtype)))
        print('Transfor entropy ' + str(entropy(rwa_output, n_clusters=n_clusters)))
    else:
        print('DIFFER')
        
    """

    