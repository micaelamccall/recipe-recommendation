import pandas as pd
import scipy.sparse as scsp
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ast import literal_eval

interactions_train = pd.read_csv("data/interactions_train_mm.csv")
interactions_train['rating'] += 1

interactions_train_w_deets = pd.read_csv("data/interactions_train_w_deets.csv")[['u', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)
interactions_test_w_deets = pd.read_csv("data/interactions_test_w_deets.csv")[['u', 'i', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)

# Create master list of all detail
deets = pd.concat([interactions_train_w_deets['deets'], interactions_test_w_deets['deets']]).drop_duplicates().reset_index(drop=True)
# Create idx for each detail
deet_id_map = pd.DataFrame(deets).reset_index().rename(columns={'index':'d'})
# Merge idx for each detail back into train and test detail data sets 
interactions_train_w_deets = interactions_train_w_deets.merge(deet_id_map, how='left', on='deets')
interactions_test_w_deets = interactions_test_w_deets.merge(deet_id_map, how='left', on='deets')
# Find average user rating for each detail
interactions_train_deets = interactions_train_w_deets.drop(columns=['deets']).groupby(['u', 'd']).mean().reset_index()

# create user-recipe matrix
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

# Create user-detail rating matrix
num_users = np.max(interactions_train['u']) + 1
num_deets = deet_id_map['d'].iloc[-1] + 1
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

# Merge back to test data to get true ratings
eval_df = eval_df.merge(interactions_test_w_deets[['u', 'i', 'rating']].drop_duplicates(), how='left', on=['u', 'i'])
eval_df = eval_df[eval_df['rating_pred'].isna() == False].drop_duplicates()

print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))

eval_df.to_csv("results/CA_CF.csv")

# 1.0786759314096774
# 0.848033259667969

sns.scatterplot(data=eval_df, x='rating', y='rating_pred')

################## Just Ingredients - experiment #################

ingr = pd.read_csv("data/pp_ingr.csv", index_col=0)
ingr = ingr.rename(columns={'id':'recipe_id'})
interactions_train = pd.read_csv("data/interactions_train_mm.csv")[['user_id', 'recipe_id', 'rating', 'u', 'i']]
interactions_test = pd.read_csv("data/interactions_test_mm.csv")[['user_id', 'recipe_id', 'rating', 'u', 'i']]
pp_recipes = pd.read_csv("data/pp_recipes.csv")

interactions_train['rating'] += 1
interactions_test['rating'] += 1

def literal_return(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val

def add_ingr_to_recipe(interactions, ingredients):
    interactions_w_deets = interactions.merge(ingredients, how='left', on='recipe_id').drop(columns=['ingredient_ids'])
    interactions_w_deets['ingredients'] = interactions_w_deets['ingredients'].apply(literal_return)
    interactions_w_deets = interactions_w_deets.explode('ingredients')
    return interactions_w_deets
    
# Add details to training data
interactions_train_w_deets = add_ingr_to_recipe(interactions_train, ingr)
interactions_train_w_deets = interactions_train_w_deets[['user_id', 'recipe_id', 'ingredients', 'rating', 'u', 'i']].drop_duplicates().reset_index(drop=True)

# Add ingredients to testing data
interactions_test_w_deets = add_ingr_to_recipe(interactions_test, ingr)
interactions_test_w_deets = interactions_test_w_deets[['user_id', 'recipe_id', 'ingredients', 'rating', 'u', 'i']].drop_duplicates().reset_index(drop=True)

# Create master list of all ingredients and id map for each 
ingredients = pd.concat([interactions_train_w_deets['ingredients'], interactions_test_w_deets['ingredients']]).drop_duplicates().reset_index(drop=True)
ingredients_id_map = pd.DataFrame(ingredients).reset_index().rename(columns={'index':'d'})

# Merge idx for each detail back into train and test detail data sets 
interactions_train_w_deets = interactions_train_w_deets.merge(ingredients_id_map, how='left', on='ingredients')
interactions_test_w_deets = interactions_test_w_deets.merge(ingredients_id_map, how='left', on='ingredients')
# Find average user rating for each detail
interactions_train_deets = interactions_train_w_deets.drop(columns=['ingredients']).groupby(['u', 'd']).mean().reset_index()

# Create user-ingredient rating matrix
num_users = np.max(interactions_train['u']) + 1
num_ingr = ingredients_id_map['d'].iloc[-1] + 1
A = scsp.csr_matrix((interactions_train_deets['rating'], (interactions_train_deets['u'], interactions_train_deets['d'])), shape=(num_users, num_ingr))


eval_df = pd.DataFrame()

for u in interactions_test_w_deets['u'].unique():
        # get similarities for that user and delete their similarity with themself
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

        pred_df = interactions_test_w_deets[interactions_test_w_deets['u'] == u]

        pred_df = pred_df.merge(deet_pred, how='left', on='d')

        pred_df = pred_df.groupby(['u', 'i'])['deet_rating_pred'].mean().reset_index().rename(columns={'deet_rating_pred':'rating_pred'})

        eval_df = pd.concat([eval_df, pred_df])


eval_df = eval_df.merge(interactions_test_w_deets[['u', 'i', 'rating']].drop_duplicates(), how='left', on=['u', 'i'])
eval_df = eval_df[eval_df['rating_pred'].isna() == False].drop_duplicates()

print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))


sns.scatterplot(data=eval_df, x='rating', y='rating_pred')
