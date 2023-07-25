import json
import pandas as pd
import scipy.sparse as scsp
import numpy as np
import seaborn as sns
from ast import literal_eval
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import data 
ingr = pd.read_csv("data/pp_ingr.csv", index_col=0)
ingr = ingr.rename(columns={'id':'recipe_id'})
interactions_train = pd.read_csv("data/interactions_train_mm.csv")[['user_id', 'recipe_id', 'rating', 'u', 'i']]
interactions_test = pd.read_csv("data/interactions_test_mm.csv")[['user_id', 'recipe_id', 'rating', 'u', 'i']]
pp_recipes = pd.read_csv("data/pp_recipes.csv")
pp_techniques = pp_recipes[['id', 'techniques']]
pp_techniques = pp_techniques.rename(columns={'id': 'recipe_id'})


# interactions_train_w_ingr = interactions_train.merge(ingr, how='inner', on='recipe_id').drop(columns=['flavors', 'ingredient_ids'])
# interactions_train_w_ingr['ingredients'] = interactions_train_w_ingr['ingredients'].apply(literal_eval)
# interactions_train_w_ingr = interactions_train_w_ingr.explode('ingredients')
# interactions_train_w_ingr = interactions_train_w_ingr.groupby(['u', 'ingredients'])['rating'].mean().reset_index()
# interactions_test_w_ingr = interactions_test.merge(ingr, how='inner', on='recipe_id').drop(columns=['flavors', 'ingredient_ids'])
# interactions_test_w_ingr['ingredients'] = interactions_test_w_ingr['ingredients'].apply(literal_eval)
# interactions_test_w_ingr = interactions_test_w_ingr.explode('ingredients')


def add_deets_to_recipe(interactions, techniques, ingredients):
    interactions_w_deets = interactions.merge(techniques, how='left', on='recipe_id').merge(ingredients, how='inner', on='recipe_id').drop(columns=['ingredient_ids'])
    interactions_w_deets['ingredients'] = interactions_w_deets['ingredients'].apply(literal_eval)
    interactions_w_deets['techniques'] = interactions_w_deets['techniques'].apply(literal_eval)
    interactions_w_deets['flavors'] = interactions_w_deets['flavors'].apply(literal_eval)
    interactions_w_deets['deets'] = interactions_w_deets['ingredients'] + interactions_w_deets['techniques'] + interactions_w_deets['flavors']
    interactions_w_deets = interactions_w_deets.explode('deets')
    return interactions_w_deets
    
interactions_train_w_deets = add_deets_to_recipe(interactions_train, pp_techniques, ingr)
interactions_train_deets = interactions_train_w_deets[['user_id', 'deets', 'rating', 'u']].drop_duplicates().reset_index(drop=True)
interactions_train_deets = interactions_train_deets.groupby(['user_id', 'deets', 'u']).mean().reset_index()
interactions_train_deets = interactions_train_deets[interactions_train_deets['deets'].str.len() > 0]


interactions_train_deets.to_csv("data/interactions_train_deets.csv")
interactions_test_w_deets = add_deets_to_recipe(interactions_test, pp_techniques, ingr)

# interactions_test_w_deets.to_csv("data/interactions_test_w_deets.csv")
interactions_test_deets = interactions_test_w_deets[['user_id', 'deets', 'rating', 'u']].drop_duplicates().reset_index(drop=True)
interactions_test_deets = interactions_test_deets.groupby(['user_id', 'deets', 'u']).mean().reset_index()
interactions_test_deets = interactions_test_deets[interactions_test_deets['deets'].str.len() > 0]
interactions_test_deets.to_csv("data/interactions_test_deets.csv")

pred_df = interactions_test_w_deets[['u', 'i', 'deets']].merge(interactions_train_w_deets[['u', 'deets', 'rating']], on=['u', 'deets'])
pred_df = pred_df.groupby(['u', 'i'])['rating'].mean().reset_index()
pred_df.columns = ['u', 'i', 'rating_pred']

eval_df = interactions_test[['u', 'i', 'rating']].merge(pred_df, how='inner', on=['i', 'u'])
print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))

eval_df.to_csv("results/model_2.csv")
