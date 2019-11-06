import numpy as np

class DataLoader():
    def __init__(self, args):
        self.train_path = args.train_path
        self.dataset = args.dataset

    def load_data(self):
        raw = np.load(self.train_path+'/'+self.dataset+'.npy')
        print('Loaded data with '+str(raw.shape[0])+'objects')



