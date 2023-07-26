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

interactions_test_w_deets = pd.read_csv("data/interactions_test_w_deets.csv")
interactions_train_w_deets = pd.read_csv("data/interactions_train_w_deets.csv")[['user_id', 'recipe_id', 'deets', 'rating', 'u', 'i']].drop_duplicates().reset_index(drop=True)
interactions_train_deets = interactions_train_w_deets.groupby(['user_id', 'deets', 'u']).mean().reset_index()

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

sim = cosine_similarity(M_norm, dense_output=False)


for u in interactions_test['u'].unique():
    if u < 4:
        print(u)
        # get similarities for that user and delete their similarity with themself
        simt = sim[u,:].toarray()
        simt = np.delete(simt, u)
        # get top 50 most similar users 
        most_similar_users = np.argpartition(simt, -10)[-10:]
        # get the ratings for those most similar users 
        similar_user_ratings = scsp.csr_matrix(M_norm[most_similar_users])
        rated_recipes = np.unique(similar_user_ratings.nonzero()[1])
        similar_user_ratings_exploded = interactions_train_w_deets[
            (interactions_train_w_deets['u'].isin(most_similar_users)) & 
            (interactions_train_w_deets['i'].isin(rated_recipes))].groupby(
                ['user_id', 'deets', 'u']).mean().reset_index()
        for recipe in interactions_test[interactions_test['u'] == 0]['i']