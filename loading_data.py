import pandas as pd 
import numpy as np 


np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
training_data = np.load('training_data1.npy')
np.load = np_load_old

print(len(training_data))

df = pd.DataFrame(training_data)

df.to_csv(r'dataset_training.csv')


