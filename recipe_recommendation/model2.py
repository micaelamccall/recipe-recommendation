import json
import pandas as pd
import scipy.sparse as scsp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.metrics import mean_squared_error, mean_absolute_error


interactions_train_w_deets = pd.read_csv("data/interactions_train_w_deets.csv")[['u', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)
interactions_test_w_deets = pd.read_csv("data/interactions_test_w_deets.csv")[['u', 'i', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)

interactions_train_deets = interactions_train_w_deets.groupby(['deets', 'u']).mean().reset_index()
interactions_train_deets = interactions_train_deets[interactions_train_deets['deets'].str.len() > 0]

# Merge training ratings of details with testing true ratings of recipes
pred_df = interactions_test_w_deets[['u', 'i', 'deets']].merge(interactions_train_deets[['u', 'deets', 'rating']], on=['u', 'deets'])
# Group on recipes to get average of detail ratings as predicted recipe rating
pred_df = pred_df.groupby(['u', 'i'])['rating'].mean().reset_index()
pred_df.columns = ['u', 'i', 'rating_pred']

# Merge back with testing data for evaluation
eval_df = interactions_test_w_deets[['u', 'i', 'rating']].merge(pred_df, how='inner', on=['i', 'u'])
eval_df = eval_df.drop_duplicates()

from model_comparisons import calculate_metrics, plot_results

true_total = len(interactions_test_w_deets[['u', 'i']].drop_duplicates())

calculate_metrics(eval_df, true_total, "CB")

eval_df.to_csv("results/CB.csv")
# 1.398715133501436
# 1.0337515990485506
# 0.35352078827204997