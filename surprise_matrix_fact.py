import numpy as np
from surprise import NMF
from surprise import Dataset, Reader, accuracy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as scsp
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

interactions_train = pd.read_csv("data/interactions_train_mm.csv", index_col=0)
interactions_test = pd.read_csv("data/interactions_test_mm.csv", index_col=0)

interactions_train['rating'] += 1
interactions_test['rating'] += 1

reader = Reader(rating_scale=(1, 6))

traindata = Dataset.load_from_df(interactions_train[["u", "i", "rating"]], reader)
traindata = traindata.build_full_trainset()

testdata = Dataset.load_from_df(interactions_test[["u", "i", "rating"]], reader)
testdata = testdata.build_full_trainset().build_testset()


algo = NMF(n_factors=40, reg_pu=0.01, reg_qi=0.01, n_epochs=1)
algo.fit(traindata)

predictions = algo.test(testdata)
# accuracy.rmse(predictions, verbose=True)

eval_df = pd.DataFrame.from_records([{'u': pred.uid, 'i': pred.iid, 'rating': pred.r_ui, 'rating_pred': pred.est} for pred in predictions])

print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))

sns.scatterplot(data=eval_df, x='rating', y='rating_pred')





