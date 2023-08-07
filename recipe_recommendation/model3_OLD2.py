import json
import pandas as pd
import scipy.sparse as scsp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error


# interactions_train = pd.read_csv("data/interactions_train_mm.csv")
# interactions_test = pd.read_csv("data/interactions_test_mm.csv")
interactions_train = pd.read_csv("data/interactions_train_deets.csv")
# interactions_test_deets = pd.read_csv("data/interactions_test_deets.csv")
# interactions_test = pd.read_csv("data/interactions_test_mm.csv")[['user_id', 'recipe_id', 'rating', 'u', 'i']]
interactions_test_w_deets = pd.read_csv("data/interactions_test_w_deets.csv")[['user_id', 'recipe_id', 'rating', 'u', 'i', 'deets']]



interactions_test_w_deets = interactions_test_w_deets[interactions_test_w_deets['deets'].isna() == False]


num_users = np.max(interactions_train['u']) + 1

deets = interactions_train['deets'].drop_duplicates().reset_index(drop=True)

deet_id_map = {}

for i in range(deets.shape[0]):
    deet_id_map[deets.loc[i,]] = i


interactions_train['i'] = interactions_train['deets'].apply(lambda x: deet_id_map[x]) 
interactions_test_w_deets['d'] = interactions_test_w_deets['deets'].apply(lambda x: deet_id_map[x]) 




num_deets = np.max(interactions_train['i']) + 1
# create user-movie matrix
# add 1 to each rating so that all nonzero elements of the sparse matrix represent a rating
M = scsp.csr_matrix((interactions_train['rating']+1, (interactions_train['u'], interactions_train['i'])))

# Calculate user level rating average
user_counts = np.diff(M.indptr).reshape(-1,1).astype(float)
user_counts[np.where(user_counts == 0)] = np.NAN
user_average = np.array(M.sum(axis=1) / user_counts)
# Create rating average sparse matrix using the locations of the nonzero elements of interactions matrix
nz = M.nonzero()
user_average = scsp.csr_matrix((user_average[nz[0]].T[0], nz))

# Calculate normalized ratings matrix
M_norm = M - user_average

sim = cosine_similarity(M_norm)


# 
# pred_matrix = scsp.csr_matrix(np.empty(shape=(0, num_recipes)))


# pred = scsp.lil_matrix((num_users, num_deets))
pred = []

for u in interactions_test_w_deets['u'].unique():
    # if u < 4:
        # print(u)
        # get similarities for that user and delete their similarity with themself
        simt = sim[u,:]
        simt = np.delete(simt, u)
        # get top 50 most similar users 
        most_similar_users = np.argpartition(simt, -10)[-10:]
        # get the ratings for those most similar users 
        similar_user_ratings = M_norm[most_similar_users]
        # calculate weighted score for each recipe (score times similarity)
        score = scsp.csc_matrix(similar_user_ratings.multiply(simt[most_similar_users].reshape(-1,1)))
        # calculate average score for each recipe
        row_idx = score.nonzero()[0]
        col_idx = score.nonzero()[1]
        col_totals = score.sum(axis=0)
        col_counts = np.diff(score.indptr).astype(float)
        col_counts[np.where(col_counts == 0)] = np.NAN
        score_average = np.array(col_totals / col_counts)[0]
        # Add in user average and subtract 1 because of adding it when making the matrix
        score_prediction = score_average + np.max(user_average[u]) - 1
        pred[(u, [i for i in range(num_deets)])] = score_prediction

        # for (u, i, rat) in zip([u] * len(col_idx), col_idx, score_prediction):
        #     pred.append((u, i, rat))
        # change type to sparse matrix
        # score_prediction = scsp.csr_matrix((score_prediction, (([0] * len(col_idx)), col_idx)), shape=(1, num_recipes))
        # add to prediction matrix
        # pred_matrix = scsp.vstack([pred_matrix, score_prediction])

pred = scsp.csr_matrix(pred)

eval_df = interactions_test_w_deets.copy()
eval_df.loc[:, 'deet_rating_pred'] = np.nan

nz = pred.nonzero()
for u, d in zip(nz[0], nz[1]):
    eval_df.loc[(eval_df['u']==u)&(eval_df['d']==d), 'deet_rating_pred'] = pred[u,d]


pred_df = eval_df.groupby(['u', 'i'])['deet_rating_pred'].mean().reset_index()
pred_df.columns = ['u', 'i', 'rating_pred']

eval_df = eval_df.merge(pred_df, how='left', on=['u', 'i']).sort_values(['u', 'i'])
eval_df = eval_df[['u', 'i', 'rating', 'rating_pred']].drop_duplicates()
eval_df = eval_df[eval_df['rating_pred'].isna() == False].drop_duplicates()


for u in pred_df['u'].unique():
    pred_df.loc[pred_df['u'] == u]


interactions_test.merge(interactions_test_deets, how='left', on)


pred_df = pd.DataFrame.from_records(pred)
pred_df.columns = ['u', 'i', 'rating_pred']


eval_df.to_csv("results/model_1.csv")

print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))


