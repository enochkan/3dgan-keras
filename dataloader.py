import numpy as np
from pathlib import Path
from scipy.io import loadmat

class DataLoader():
    def __init__(self, args):
        self.train_path = args.train_path
        self.dataset = args.dataset

    def load_data(self):
        if self.dataset == 'ikea':
            X_train = []
            for file in Path('./data/').glob('*.mat'):
                print('Loading volume... ' + str(file))
                data = loadmat(str(file))
                X_train.append(data['voxel'])
            return X_train
            # get file length
        # raw = np.load(self.train_path+'/'+self.dataset+'.mat')
        # print('Loaded data with '+str(raw.shape[0])+'objects')




