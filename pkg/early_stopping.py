import numpy as np

class EarlyStopping():
    def __init__(self, patience):
        self.min_acc = 0
        self.counter = 0
        self.patience = patience

    def early_stop(self, val_acc):
        if val_acc >= self.min_acc:
            self.min_acc = val_acc
            self.counter = 0
        else:
            self.counter +=1
            if self.counter >= self.patience:
                return True