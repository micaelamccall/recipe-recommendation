import json
import pandas as pd
import scipy.sparse as scsp
import numpy as np
import seaborn as sns
from ast import literal_eval
from sklearn.metrics import mean_squared_error, mean_absolute_error


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

interactions_test_w_deets = add_ingr_to_recipe(interactions_test, ingr)
interactions_test_w_deets = interactions_test_w_deets[['user_id', 'recipe_id', 'ingredients', 'rating', 'u', 'i']].drop_duplicates().reset_index(drop=True)


interactions_train_deets = interactions_train_w_deets.groupby(['ingredients', 'u']).mean().reset_index()
interactions_train_deets = interactions_train_deets[interactions_train_deets['ingredients'].str.len() > 0]

# Merge training ratings of details with testing true ratings of recipes
pred_df = interactions_test_w_deets[['u', 'i', 'ingredients']].merge(interactions_train_deets[['u', 'ingredients', 'rating']], on=['u', 'ingredients'])
# Group on recipes to get average of detail ratings as predicted recipe rating
pred_df = pred_df.groupby(['u', 'i'])['rating'].mean().reset_index()
pred_df.columns = ['u', 'i', 'rating_pred']

# Merge back with testing data for evaluation
eval_df = interactions_test_w_deets[['u', 'i', 'rating']].merge(pred_df, how='inner', on=['i', 'u'])
eval_df = eval_df.drop_duplicates()
print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))
# 1.0146114345223411
# 0.5379449783796681

eval_df.to_csv("results/model_2.csv")
sns.scatterplot(data=eval_df, x='rating', y='rating_pred')
