import pandas as pd
import os

print(os.getcwd())
path = os.path.join(os.getcwd(), 'data/num_image_modalities.csv')

data = pd.read_csv(path)
print(data)
testdata = data.sample(frac = 0.1)
traindata = data.drop(testdata.index)

testdata.sort_index(inplace=True)
traindata.sort_index(inplace=True)

testdata.to_csv('data/testsplit_ids.csv')
traindata.to_csv('data/trainsplit_ids.csv')