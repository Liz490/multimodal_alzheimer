import numpy as np
import torch
import seaborn as sn
import matplotlib.pyplot as plt


path = '/vol/chameleon/projects/adni/adni_1/trained_models/pet1451_epoch25.pth'

checkpoint = torch.load(path)

confusion = checkpoint['confusion']
frequency = np.sum(confusion, axis=1)
print(frequency)
print(confusion)
axis_labels = ['CN', 'MCI', 'AD']
plt.figure(figsize = (10,7))

sn.set(font_scale=1.4)
sn.heatmap(confusion, annot=True, xticklabels=axis_labels, yticklabels=axis_labels, cmap= "Reds")
plt.xlabel("Prediction", fontsize=17)
plt.ylabel("Label", fontsize=17)
plt.xticks(fontsize=15)#, rotation=90)
plt.yticks(fontsize=15)
plt.savefig('/vol/chameleon/projects/adni/adni_1/figures/confusion.png')

