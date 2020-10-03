import pandas as pd
from scipy.io import loadmat

def readmat(init_data_file, real_data_file):
    #读取mat文件,并且转换成dataframe的格式
    #mat_x = pd.DataFrame(loadmat("./data/x.mat")["x"])
    mat_y1 = pd.DataFrame(loadmat(init_data_file)["y1"])
    mat_y2 = pd.DataFrame(loadmat(real_data_file)["y2"])

    if (mat_y1.shape[1] == mat_y2.shape[1]):
        longth = mat_y1.shape[1]

        f = open("./data/pathfile.txt", "w")

        #将pd文件转成csv文件
        # for i in range(longth):
        for i in range(10):
            temp_init = mat_y1[i]
            filename_inital = "./data/csv/init"+str(i)+".csv"
            temp_init.to_csv(filename_inital, index = False, sep=',')
            temp_real = mat_y2[i]
            filename_real = "./data/csv/real" + str(i) + ".csv"
            temp_real.to_csv(filename_real, index=False, sep=',')
            f.write(filename_inital+' '+filename_real+'\r')

    return longth


# path_y1 = "./data/y1.mat"
# path_y2 = "./data/y2.mat"
# readmat(path_y1, path_y2)