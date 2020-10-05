from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split

'''
用于读取光谱文件并且进行处理
'''

def readmat(init_data_file, real_data_file):
    # 读取mat文件,并且转换成ndarray
    a = loadmat(init_data_file)
    b = loadmat(real_data_file)
    data_impure = a['data']
    data_pure = b['data']

    X = data_pure[0, 0]['data_x']
    Y = data_impure[0, 0]['data_y']

    num_raws = len(X[0])
    num_lines = len(X)
    X = np.array(X).reshape(num_lines, num_raws).astype('float32')
    Y = np.array(Y).reshape(num_lines, num_raws).astype('float32')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train = np.expand_dims(X_train, 1)
    Y_train = np.expand_dims(Y_train, 1)
    X_test = np.expand_dims(X_test, 1)
    Y_test = np.expand_dims(Y_test, 1)
    return X_train,Y_train,X_test,Y_test


# path_y1 = "./data/data_impure.mat"
# path_y2 = "./data/data_pure.mat"
# readmat(path_y1, path_y2)