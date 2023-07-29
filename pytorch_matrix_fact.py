import numpy as np
import pandas as pd
import seaborn as sns
import scipy.sparse as scsp
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

interactions_train = pd.read_csv("data/interactions_train_small_mm.csv")
interactions_test = pd.read_csv("data/interactions_test_small_mm.csv")

num_users = np.max(interactions_train['u']) + 1
num_recipes = np.max(interactions_train['i']) + 1
# create user-recipe matrix
# add 1 to each rating so that all nonzero elements of the sparse matrix represent a rating

R = scsp.csr_matrix((interactions_train['rating'], (interactions_train['u'], interactions_train['i'])))

n_factors = 40
num_epochs = 200
lr = 0.1

# Create random initial latent matrices
U = scsp.csr_matrix(np.random.uniform(-1, 1, size=(num_users, n_factors)))
V = scsp.csr_matrix(np.random.uniform(-1, 1, size=(num_recipes, n_factors)))
N = num_users * num_recipes

class PytorchLinearModel(torch.nn.Module):
    def __init__(self, num_users, num_items, K, lr):
        super().__init__()
        # Set U and V as parameters of this model
        self.U = torch.nn.Parameter(torch.zeros((num_users, K)))
        self.V = torch.nn.Parameter(torch.zeros((num_items, K)))
        # Xavier initialization is a great way to intialize parameters
        torch.nn.init.xavier_uniform_(self.U)
        torch.nn.init.xavier_uniform_(self.V)
        # MSE using Pytorch
        self.MSE = torch.nn.MSELoss()
        # Optimizer for handling the gradient calculation and parameter updates
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
    
    def forward(self):
        return self.U @ self.V.T
    
    def calculate_loss(self, R, R_hat):
        return self.MSE(R, R_hat)
    
    def optimize(self, loss):
        # Send in the loss tensor from `calculate_loss` to update parameters
        self.optimizer.zero_grad()  # Clear any previous epoch gradients
        loss.backward()  # Calc gradient
        self.optimizer.step()  # Update parameters
        
# Create Model -> U and V
MF_model = PytorchLinearModel(num_users, num_recipes, n_factors, lr)

# row = torch.from_numpy(R.row.astype(np.int64)).to(torch.long)
# col = torch.from_numpy(R.col.astype(np.int64)).to(torch.long)
# edge_index = torch.stack([row, col], dim=0)
# val = torch.from_numpy(R.data.astype(np.float64)).to(torch.float)
# R_tens = torch.sparse_coo_tensor(edge_index, val, torch.Size(R.shape))
# R_tens.to_dense()

R_tens = torch.FloatTensor(R.todense(), )

# R_tens = torch.from_numpy(R.todense()).to_sparse()

# Training
for curr_epoch in range(num_epochs):
    # Reconstruct R_hat from latent factor matrices
    R_hat = MF_model.forward()
    # Calc MSE loss of this reconstruction
    loss = MF_model.calculate_loss(R_tens, R_hat)
    print(loss)
    # Calc grad and update
    MF_model.optimize(loss)

    
eval_df = interactions_test[['u', 'i', 'rating']]
eval_df.loc[:, 'rating_pred'] = np.nan

for (u, i) in zip(interactions_test['u'], interactions_test['i']):
    pred= MF_model.U[u].dot(MF_model.V[i].T).item()
    eval_df.loc[eval_df['u']==u, 'rating_pred'] = pred

global_mean = np.mean(interactions_train['rating'])

eval_df['rating_pred'] += global_mean
print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))

sns.scatterplot(data=eval_df, x='rating', y='rating_pred')

