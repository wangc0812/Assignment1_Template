import numpy as np
import struct

def load_mnist():
    with open('./data/t10k-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows, ncols))/255
        test_data = test_data[:,:,:,np.newaxis]
    with open('./data/t10k-labels.idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_label = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_label = test_label.reshape((size,))

    with open('./data/train-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows, ncols))/255
        train_data = train_data[:,:,:,np.newaxis]
    with open('./data/train-labels.idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_label = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_label = train_label.reshape((size,))

    return train_data,train_label,test_data, test_label
