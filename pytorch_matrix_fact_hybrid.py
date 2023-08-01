import numpy as np
import pandas as pd
import seaborn as sns
import scipy.sparse as scsp
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from ast import literal_eval

interactions_train = pd.read_csv("data/interactions_train_small_mm.csv")
interactions_test = pd.read_csv("data/interactions_test_small_mm.csv")
pp_recipes = pd.read_csv("data/pp_recipes.csv")
ingr = pd.read_csv("data/pp_ingr.csv", index_col=0)
ingr = ingr.rename(columns={'id':'recipe_id'})

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

# Add details to testing data
interactions_test_w_deets = add_ingr_to_recipe(interactions_test, ingr)
interactions_test_w_deets = interactions_test_w_deets[['user_id', 'recipe_id', 'ingredients', 'rating', 'u', 'i']].drop_duplicates().reset_index(drop=True)


interactions_w_deets = pd.concat([interactions_train_w_deets, interactions_test_w_deets])

# Create full list of ingredients and find unique id for each
ingredients = interactions_w_deets['ingredients'].drop_duplicates().reset_index(drop=True)
ingredients_id_map = pd.DataFrame(ingredients).reset_index().rename(columns={'index':'d'})

# Add ids for ingredients in
recipe_ingredients = interactions_w_deets.merge(ingredients_id_map, how='left', on='ingredients')[['d', 'i']].drop_duplicates()
recipe_ingredients_train = interactions_train_w_deets.merge(ingredients_id_map, how='left', on='ingredients')[['d', 'i']].drop_duplicates()
recipe_ingredients_test = interactions_test_w_deets.merge(ingredients_id_map, how='left', on='ingredients')[['d', 'i']].drop_duplicates()

num_users = np.max(interactions_train['u']) + 1
num_recipes = np.max(interactions_train['i']) + 1
num_ingr = ingredients_id_map['d'].iloc[-1] + 1

# create user-recipe matrix
R = scsp.csr_matrix((interactions_train['rating'], (interactions_train['u'], interactions_train['i'])))
X = scsp.csr_matrix((np.ones((len(recipe_ingredients))), (recipe_ingredients['i'], recipe_ingredients['d'])))
X_train = scsp.csr_matrix((np.ones((len(recipe_ingredients_train))), (recipe_ingredients_train['i'], recipe_ingredients_train['d'])))
X_test = scsp.csr_matrix((np.ones((len(recipe_ingredients_test))), (recipe_ingredients_test['i'], recipe_ingredients_test['d'])))

n_factors = 40
num_epochs = 200
lr = 1.5

class PytorchLinearModel(torch.nn.Module):
    def __init__(self, num_users, num_ingr, K, lr):
        super().__init__()
        # Set U and V as parameters of this model
        self.U = torch.nn.Parameter(torch.zeros((num_users, K)))
        self.PHI = torch.nn.Parameter(torch.zeros((num_ingr, K)))
        # Xavier initialization is a great way to intialize parameters
        torch.nn.init.xavier_uniform_(self.U)
        torch.nn.init.xavier_uniform_(self.PHI)

        # torch.nn.init.uniform_(self.U, 0, 1)
        # torch.nn.init.uniform_(self.PHI, 0, 1)
        # MSE using Pytorch
        self.MSE = torch.nn.MSELoss()
        # Optimizer for handling the gradient calculation and parameter updates
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
    
    def forward(self, X):
        return self.U @ self.PHI.T @ X.T
    
    def calculate_loss(self, R, R_hat):
        return self.MSE(R, R_hat)
    
    def optimize(self, loss):
        # Send in the loss tensor from `calculate_loss` to update parameters
        self.optimizer.zero_grad()  # Clear any previous epoch gradients
        loss.backward()  # Calc gradient
        self.optimizer.step()  # Update parameters
        
# Create Model -> U and V
MF_model = PytorchLinearModel(num_users, num_ingr, n_factors, lr)

R_tens = torch.FloatTensor(R.todense())
X_tens = torch.FloatTensor(X.todense())
# X_test_tens = torch.FloatTensor(X_test.todense())
# Training
losses = []
for curr_epoch in range(num_epochs):
    # Reconstruct R_hat from latent factor matrices
    R_hat = MF_model.forward(X_tens)
    # Calc MSE loss of this reconstruction
    loss = MF_model.calculate_loss(R_tens, R_hat)
    if curr_epoch % 5 == 0:
        print(f"Epoch: {curr_epoch}")
        print(f"Loss: {loss.item()}")
        losses.append(loss.item())
    # Calc grad and update
    MF_model.optimize(loss)

    
eval_df = interactions_test[['u', 'i', 'rating']]
eval_df.loc[:, 'rating_pred'] = np.nan

for (u, i) in zip(interactions_test['u'], interactions_test['i']):
    pred = (MF_model.U[u] @ MF_model.PHI.T).dot(X_tens[i]).item()
    eval_df.loc[eval_df['u']==u, 'rating_pred'] = pred

global_mean = np.mean(interactions_train['rating'])

eval_df['rating_pred'] += global_mean

eval_df.loc[eval_df['rating_pred'] > 6, 'rating_pred'] = 6
eval_df.loc[eval_df['rating_pred'] < 1, 'rating_pred'] = 1



print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))

sns.scatterplot(data=eval_df, x='rating', y='rating_pred')

