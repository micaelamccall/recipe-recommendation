import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

interactions_train = pd.read_csv("data/interactions_train_mm.csv", index_col=0)
interactions_test = pd.read_csv("data/interactions_test_mm.csv", index_col=0)

interactions_train['rating'] += 1
interactions_test['rating'] += 1

### Random choice with the probabilities with which the different ratings appear in the training data
p = []
for r in range(1, 7):
    p.append(len(interactions_train[interactions_train['rating'] == r]) / len(interactions_train))

rating_pred = np.random.choice(np.arange(1, 7), len(interactions_test), p=p)

### Random choice with from a unform distribution from 1 to 7
rating_pred = np.random.uniform(1, 7, len(interactions_test))

eval_df = interactions_test[['u', 'i', 'rating']]
eval_df.loc[:, 'rating_pred'] = rating_pred
