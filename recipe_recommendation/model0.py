import pandas as pd
import scipy.sparse as scsp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

interactions_train = pd.read_csv("data/interactions_train_mm.csv", index_col=0)
interactions_test = pd.read_csv("data/interactions_test_mm.csv", index_col=0)

interactions_train['rating'] += 1
interactions_test['rating'] += 1

num_users = np.max(interactions_train['u']) + 1
num_recipes = np.max(interactions_train['i']) + 1

p = []
for r in range(1, 7):
    p.append(len(interactions_train[interactions_train['rating'] == r]) / len(interactions_train))

rating_pred = np.random.choice(np.arange(1, 7), len(interactions_test), p=p)

rating_pred = np.random.uniform(1, 7, len(interactions_test))

eval_df = interactions_test[['u', 'i', 'rating']]
eval_df.loc[:, 'rating_pred'] = rating_pred


# eval_df = interactions_test[['u', 'i', 'rating']].merge(pred_df, how='left', on=['i', 'u'])
eval_df = eval_df[eval_df['rating_pred'].isna() == False].drop_duplicates()

eval_df.to_csv("results/bl_model.csv")

print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))


sns.scatterplot(data=eval_df, x='rating', y='rating_pred')


sns.set_theme(style="whitegrid")
g = sns.FacetGrid(eval_df, col='rating', col_wrap=3)
g.map(sns.histplot, 'rating_pred')