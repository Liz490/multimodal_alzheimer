import numpy as np

class EarlyStopping():
    def __init__(self, patience):
        self.min_loss = np.inf
        self.counter = 0
        self.patience = patience

    def early_stop(self, val_loss):
        if val_loss <= self.min_loss:
            self.min_loss = val_loss
            self.counter = 0
        else:
            self.counter +=1
            if self.counter >= self.patience:
                return True