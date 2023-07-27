import json
import pandas as pd
import scipy.sparse as scsp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as spst
from sklearn.metrics import mean_squared_error, mean_absolute_error

interactions_train = pd.read_csv("data/interactions_train_mm.csv")
interactions_test = pd.read_csv("data/interactions_test_mm.csv")

interactions_test_w_deets = pd.read_csv("data/interactions_test_w_deets.csv")[['u', 'i', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)
interactions_train_w_deets = pd.read_csv("data/interactions_train_w_deets.csv")[['u', 'i', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)

deets = pd.concat([interactions_train_w_deets['deets'], interactions_test_w_deets['deets']]).drop_duplicates().reset_index(drop=True)

deet_id_map = pd.DataFrame(deets).reset_index().rename(columns={'index':'d'})

interactions_train_w_deets = interactions_train_w_deets.merge(deet_id_map, how='left', on='deets')
interactions_test_w_deets = interactions_test_w_deets.merge(deet_id_map, how='left', on='deets')


interactions_train_deets = interactions_train_w_deets.drop(columns=['deets']).groupby(['u', 'd']).mean().reset_index()

num_users = np.max(interactions_train['u']) + 1
num_recipes = np.max(interactions_train['i']) + 1
# create user-recipe matrix
# add 1 to each rating so that all nonzero elements of the sparse matrix represent a rating

M = scsp.csr_matrix((interactions_train['rating']+1, (interactions_train['u'], interactions_train['i'])))

# Calculate user level rating average
user_average = np.array(M.sum(axis=1) / np.diff(M.indptr).reshape(-1,1))
# Create rating average sparse matrix using the locations of the nonzero elements of interactions matrix
nz = M.nonzero()
user_average = scsp.csr_matrix((user_average[nz[0]].T[0], nz))

# Calculate normalized ratings matrix
M_norm = M - user_average
# Calculate similarity matrix
sim = cosine_similarity(M_norm, dense_output=False)

A = scsp.csr_matrix((interactions_train_deets['rating']+1, (interactions_train_deets['u'], interactions_train_deets['d'])))

eval_df = pd.DataFrame()

for u in interactions_test_w_deets['u'].unique():
    if u < 5:
        print(u)
        # get similarities for that user and delete their similarity with themself
        simt = sim[u,:].toarray()
        simt = np.delete(simt, u)
        # get top 50 most similar users 
        most_similar_users = np.argpartition(simt, -10)[-10:]
        # get the ratings for those most similar users 
        # similar_user_ratings = scsp.csr_matrix(M_norm[most_similar_users])
        # rated_recipes = np.unique(similar_user_ratings.nonzero()[1])
        # similar_user_deet_ratings = interactions_train_deets[
        #     interactions_train_deets['u'].isin(most_similar_users)]
        similar_user_deet_ratings = scsp.csr_matrix(A[most_similar_users])
        
        deet_score = scsp.csc_matrix(similar_user_deet_ratings.multiply(simt[most_similar_users].reshape(-1,1)) )
        deet_pred = deet_score.sum(axis=0) / simt[most_similar_users].sum()
        deet_pred[np.where(deet_pred > 0)] = deet_pred[deet_pred > 0]  + 1
        deet_pred = pd.DataFrame(deet_pred.T).reset_index().rename(columns={'index': 'd', 0: 'deet_rating_pred'})
        # deet_pred['u'] = u

        pred_df = interactions_test_w_deets[interactions_test_w_deets['u'] == u]

        pred_df = pred_df.merge(deet_pred, how='left', on='d')

        pred_df = pred_df.groupby(['u', 'i'])['deet_rating_pred'].mean().reset_index().rename(columns={'deet_rating_pred':'rating_pred'})

        eval_df = pd.concat([eval_df, pred_df])



eval_df.merge(interactions_test_w_deets[['u', 'i', 'rating']].drop_duplicates(), how='left', on=['u', 'i'])