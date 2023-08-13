import numpy as np
from surprise import NMF
from surprise import Dataset, Reader, accuracy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as scsp      
from sklearn.metrics import mean_squared_error, mean_absolute_error


interactions_train = pd.read_csv("data/interactions_train_mm.csv", index_col=0)
interactions_test = pd.read_csv("data/interactions_test_mm.csv", index_col=0)

interactions_train['rating'] += 1
interactions_test['rating'] += 1

num_users = np.max(interactions_train['u']) + 1
num_recipes = np.max(interactions_train['i']) + 1

n_factors = 40
num_epochs = 5
lr = 0.1

# auxiliary matrices used in optimization process
user_num = np.zeros((num_users, n_factors))
user_denom = np.zeros((num_users, n_factors))
item_num = np.zeros((num_recipes, n_factors))
item_denom = np.zeros((num_recipes, n_factors))

all_ratings = zip(interactions_train['u'], interactions_train['i'], interactions_train['rating'])
reg_pu=.06; reg_qi=.06
global_mean = np.mean(interactions_train['rating'])

pu = np.random.uniform(0, 1, size=(num_users, n_factors))
qi = np.random.uniform(0, 1, size=(num_recipes, n_factors))

for current_epoch in range(num_epochs):
    user_num[:, :] = 0
    user_denom[:, :] = 0
    item_num[:, :] = 0
    item_denom[:, :] = 0
    all_ratings = zip(interactions_train['u'], interactions_train['i'], interactions_train['rating'])

    # Compute numerators and denominators for users and items factors
    for u, i, r in all_ratings:
        # compute current estimation and error
        dot = 0  # <q_i, p_u>
        for f in range(n_factors):
            dot += qi[i, f] * pu[u, f]
        est = 0 + dot
        err = r - est

        # compute numerators and denominators
        for f in range(n_factors):
            user_num[u, f] += qi[i, f] * r
            user_denom[u, f] += qi[i, f] * est
            item_num[i, f] += pu[u, f] * r
            item_denom[i, f] += pu[u, f] * est

    # Update user factors
    for u in interactions_train['u'].drop_duplicates():
        n_ratings = len(interactions_train[interactions_train['u'] == u].drop_duplicates())
        for f in range(n_factors):
            if pu[u, f] != 0:  # Can happen if user only has 0 ratings
                user_denom[u, f] += n_ratings * reg_pu * pu[u, f]
                pu[u, f] *= user_num[u, f] / user_denom[u, f]

    # Update item factors
    for i in interactions_train['i'].drop_duplicates():
        n_ratings = len(interactions_train[interactions_train['i']==i].drop_duplicates())
        for f in range(n_factors):
            if qi[i, f] != 0:
                item_denom[i, f] += n_ratings * reg_qi * qi[i, f]
                # print(item_num[i, f])
                qi[i, f] *= item_num[i, f] / item_denom[i, f]

pu = np.asarray(pu)
qi = np.asarray(qi)

eval_df = interactions_test[['u', 'i', 'rating']]
eval_df.loc[:, 'rating_pred'] = np.nan

for (u, i) in zip(interactions_test['u'], interactions_test['i']):
    pred= pu[u].dot(qi[i].T).item()
    eval_df.loc[eval_df['u']==u, 'rating_pred'] = pred
eval_df.loc[eval_df['rating_pred'] > 6, 'rating_pred'] = 6

print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))

sns.scatterplot(data=eval_df, x='rating', y='rating_pred')

