import pandas as pd
import scipy.sparse as scsp
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

interactions_train = pd.read_csv("data/interactions_train_mm.csv")

interactions_train['rating'] += 1

interactions_train_w_deets = pd.read_csv("data/interactions_train_w_deets.csv")[['u', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)
interactions_test_w_deets = pd.read_csv("data/interactions_test_w_deets.csv")[['u', 'i', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)

deets = pd.concat([interactions_train_w_deets['deets'], interactions_test_w_deets['deets']]).drop_duplicates().reset_index(drop=True)

deet_id_map = pd.DataFrame(deets).reset_index().rename(columns={'index':'d'})

interactions_train_w_deets = interactions_train_w_deets.merge(deet_id_map, how='left', on='deets')
interactions_test_w_deets = interactions_test_w_deets.merge(deet_id_map, how='left', on='deets')


interactions_train_deets = interactions_train_w_deets.drop(columns=['deets']).groupby(['u', 'd']).mean().reset_index()

num_users = np.max(interactions_train['u']) + 1
num_recipes = np.max(interactions_train['i']) + 1
num_deets = deet_id_map['d'].iloc[-1] + 1
# create user-recipe matrix
# add 1 to each rating so that all nonzero elements of the sparse matrix represent a rating

M = scsp.csr_matrix((interactions_train['rating'], (interactions_train['u'], interactions_train['i'])))

# Calculate user level rating average
user_average = np.array(M.sum(axis=1) / np.diff(M.indptr).reshape(-1,1))
# Create rating average sparse matrix using the locations of the nonzero elements of interactions matrix
nz = M.nonzero()
user_average = scsp.csr_matrix((user_average[nz[0]].T[0], nz))

# Calculate normalized ratings matrix
M_norm = M - user_average
# Calculate similarity matrix
sim = cosine_similarity(M_norm, dense_output=False)

A = scsp.csr_matrix((interactions_train_deets['rating'], (interactions_train_deets['u'], interactions_train_deets['d'])), shape=(num_users, num_deets))

eval_df = pd.DataFrame()

for u in interactions_test_w_deets['u'].unique():
    # if u < 6:
    #     print(u)
        simt = sim[u,:].toarray()[0]
        simt = np.delete(simt, u)

        # get top 50 most similar users 
        most_similar_users = np.argpartition(simt, -50)[-50:]
        # get the ratings for those most similar users 
        similar_user_deet_ratings = scsp.csc_matrix(A[most_similar_users])
        
        # deet_score = scsp.csc_matrix(similar_user_deet_ratings.multiply(simt[most_similar_users].reshape(-1,1)) )
        col_counts = np.diff(similar_user_deet_ratings.indptr).astype(float)
        col_counts[np.where(col_counts == 0)] = np.NAN
        deet_pred = similar_user_deet_ratings.sum(axis=0) / col_counts
        # deet_pred[np.where(deet_pred > 0)] = deet_pred[deet_pred > 0] 
        deet_pred = pd.DataFrame(deet_pred.T).reset_index().rename(columns={'index': 'd', 0: 'deet_rating_pred'})
        # deet_pred['actual'] = A[u].todense().T
        # deet_pred['deet_rating_pred'] = np.where(deet_pred['actual'] > 0, deet_pred['actual'], deet_pred['deet_rating_pred'])

        pred_df = interactions_test_w_deets[interactions_test_w_deets['u'] == u]
        pred_df = pred_df.merge(deet_pred, how='left', on='d')
        pred_df = pred_df.groupby(['u', 'i'])['deet_rating_pred'].mean().reset_index().rename(columns={'deet_rating_pred':'rating_pred'})

        eval_df = pd.concat([eval_df, pred_df])



eval_df = eval_df.merge(interactions_test_w_deets[['u', 'i', 'rating']].drop_duplicates(), how='left', on=['u', 'i'])
eval_df = eval_df[eval_df['rating_pred'].isna() == False].drop_duplicates()

print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))

eval_df.to_csv("results/CA_CF.csv")

# 1.0786759314096774
# 0.848033259667969

sns.scatterplot(data=eval_df, x='rating', y='rating_pred')
