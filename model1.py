import json
import pandas as pd
import scipy.sparse as scsp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as spst
from sklearn.metrics import mean_squared_error, mean_absolute_error

ingr = pd.read_csv("data/pp_ingr.csv", index_col=0)
ingr = ingr.rename(columns={'id':'recipe_id'})
interactions_train = pd.read_csv("data/interactions_train_mm.csv")
interactions_test = pd.read_csv("data/interactions_test_mm.csv")

# interactions_test[interactions_test['user_id'] == 1535]
# g = interactions_test.groupby('u').nunique()['i']

num_users = np.max(interactions_train['u']) + 1
num_recipes = np.max(interactions_train['i']) + 1
# create user-movie matrix
# add 1 to each rating so that all nonzero elements of the sparse matrix represent a rating

M = scsp.csr_matrix((interactions_train['rating']+1, (interactions_train['u'], interactions_train['i'])))

# Calculate user level rating average
user_average = np.array(M.sum(axis=1) / np.diff(M.indptr).reshape(-1,1))
# Create rating average sparse matrix using the locations of the nonzero elements of interactions matrix
nz = M.nonzero()
user_average = scsp.csr_matrix((user_average[nz[0]].T[0], nz))

# Calculate normalized ratings matrix
M_norm = M - user_average

sim = cosine_similarity(M_norm)


# 
# pred_matrix = scsp.csr_matrix(np.empty(shape=(0, num_recipes)))


pred = []

for u in interactions_test['u'].unique():
    # if u < 3:
        # print(u)
        # get similarities for that user and delete their similarity with themself
        simt = sim[u,:]
        simt = np.delete(simt, u)
        # get top 50 most similar users 
        most_similar_users = np.argpartition(simt, -50)[-50:]
        # get the ratings for those most similar users 
        similar_user_ratings = M[most_similar_users]
        # calculate weighted score for each recipe (score times similarity)
        score = scsp.csc_matrix(similar_user_ratings.multiply(simt[most_similar_users].reshape(-1,1)) )
        # calculate average score for each recipe
        row_idx = score.nonzero()[0]
        col_idx = score.nonzero()[1]
        col_totals = score.sum(axis=0)[:, col_idx]
        col_counts = [len(col_idx[np.where(col_idx == c)]) for c in col_idx]
        score_average = np.array(col_totals / col_counts)[0]
        # Add in user average and subtract 1 because of adding it when making the matrix
        score_prediction = score_average + np.max(user_average[u]) - 2

        for (u, i, rat) in zip([u] * len(col_idx), col_idx, score_prediction):
            pred.append((u, i, rat))
        # change type to sparse matrix
        # score_prediction = scsp.csr_matrix((score_prediction, (([0] * len(col_idx)), col_idx)), shape=(1, num_recipes))
        # add to prediction matrix
        # pred_matrix = scsp.vstack([pred_matrix, score_prediction])


pred_df = pd.DataFrame.from_records(pred)
pred_df.columns = ['u', 'i', 'rating_pred']

eval_df = interactions_test[['u', 'i', 'rating']].merge(pred_df, how='left', on=['i', 'u'])
eval_df = eval_df[eval_df['rating_pred'].isna() == False].drop_duplicates()

eval_df.to_csv("results/model_1.csv")

print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))


np.max(user_average[11433])
interactions_train[interactions_train['u'] == 11433]['rating']
np.average(interactions_train[interactions_train['u'] == 11433]['rating'])
# interactions_train[interactions_train['u'] == 22095]

# interactions_test['u'][0]
# interactions_train['user_id'][interactions_train['u'] == 2]
# len(interactions_test['user_id'][interactions_test['u'] == 2])


# for u in interactions_test['u']:
#     if len(interactions_train['user_id'][interactions_train['u'] == u]) == 0:
#         print(u)
#         # print(interactions_train['user_id'][interactions_train['u'] == u])

# np.any(interactions_test['recipe_id'].isin(interactions_train['recipe_id']))


# i = 61982
# interactions_train[(interactions_train['u'].isin(most_similar_users)) & (interactions_train['i'] == i)]

# M_norm[nz] -= user_average[nz[0]].T

# interactions_train.groupby(["user_id", "u"]).count().reset_index().sort_values(by="u")

# interactions_train.isna().sum()

