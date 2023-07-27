import json
import pandas as pd
import scipy.sparse as scsp
import numpy as np
import seaborn as sns
from ast import literal_eval
from sklearn.metrics import mean_squared_error, mean_absolute_error




# interactions_train_w_ingr = interactions_train.merge(ingr, how='inner', on='recipe_id').drop(columns=['flavors', 'ingredient_ids'])
# interactions_train_w_ingr['ingredients'] = interactions_train_w_ingr['ingredients'].apply(literal_eval)
# interactions_train_w_ingr = interactions_train_w_ingr.explode('ingredients')
# interactions_train_w_ingr = interactions_train_w_ingr.groupby(['u', 'ingredients'])['rating'].mean().reset_index()
# interactions_test_w_ingr = interactions_test.merge(ingr, how='inner', on='recipe_id').drop(columns=['flavors', 'ingredient_ids'])
# interactions_test_w_ingr['ingredients'] = interactions_test_w_ingr['ingredients'].apply(literal_eval)
# interactions_test_w_ingr = interactions_test_w_ingr.explode('ingredients')


interactions_train_w_deets = pd.read_csv("data/interactions_train_w_deets.csv")[['u', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)
interactions_test_w_deets = pd.read_csv("data/interactions_test_w_deets.csv")[['u', 'i', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)

interactions_train_deets = interactions_train_w_deets.groupby(['user_id', 'deets', 'u']).mean().reset_index()
interactions_train_deets = interactions_train_deets[interactions_train_deets['deets'].str.len() > 0]
interactions_train_deets.to_csv("data/interactions_train_deets.csv")

# Add details to testing data
interactions_test_deets = interactions_test_w_deets[['deets', 'rating', 'u']].drop_duplicates().reset_index(drop=True)
interactions_test_deets = interactions_test_deets.groupby(['deets', 'u']).mean().reset_index()
interactions_test_deets = interactions_test_deets[interactions_test_deets['deets'].str.len() > 0]
interactions_test_deets.to_csv("data/interactions_test_deets.csv")

# Merge training ratings of details with testing true ratings of recipes
pred_df = interactions_test_w_deets[['u', 'i', 'deets']].merge(interactions_train_w_deets[['u', 'deets', 'rating']], on=['u', 'deets'])
# Group on recipes to get average of detail ratings as predicted recipe rating
pred_df = pred_df.groupby(['u', 'i'])['rating'].mean().reset_index()
pred_df.columns = ['u', 'i', 'rating_pred']

# Merge back with testing data for evaluation
eval_df = interactions_test[['u', 'i', 'rating']].merge(pred_df, how='inner', on=['i', 'u'])
print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))

eval_df.to_csv("results/model_2.csv")


sns.scatterplot(data=eval_df, x='rating', y='rating_pred')
