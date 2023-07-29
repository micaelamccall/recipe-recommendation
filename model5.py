import numpy as np
from surprise import NMF
from surprise import Dataset, Reader, accuracy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as scsp
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch

interactions_train = pd.read_csv("data/interactions_train_mm.csv", index_col=0)
interactions_test = pd.read_csv("data/interactions_test_mm.csv", index_col=0)

interactions_train['rating'] += 1
interactions_test['rating'] += 1

# reader = Reader(rating_scale=(1, 6))

# traindata = Dataset.load_from_df(interactions_train[["u", "i", "rating"]], reader)
# traindata = traindata.build_full_trainset()

# testdata = Dataset.load_from_df(interactions_test[["u", "i", "rating"]], reader)
# testdata = testdata.build_full_trainset().build_testset()

# len(testdata)

# algo = NMF(n_factors=40, reg_pu=0.01, reg_qi=0.01)
# algo.fit(traindata)

# predictions = algo.test(testdata)
# # accuracy.rmse(predictions, verbose=True)

# eval_df = pd.DataFrame.from_records([{'u': pred.uid, 'i': pred.iid, 'rating': pred.r_ui, 'rating_pred': pred.est} for pred in predictions])

# print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
# print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))

# sns.scatterplot(data=eval_df, x='rating', y='rating_pred')




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
# U = scsp.csr_matrix(np.random.uniform(-1, 1, size=(num_users, n_factors)))
# V = scsp.csr_matrix(np.random.uniform(-1, 1, size=(num_recipes, n_factors)))
# N = num_users * num_recipes

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


print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))

sns.scatterplot(data=eval_df, x='rating', y='rating_pred')












class ExplicitMF:
    """
    Train a matrix factorization model using Alternating Least Squares
    to predict empty entries in a matrix
    
    Parameters
    ----------
    n_iters : int
        number of iterations to train the algorithm
        
    n_factors : int
        number of latent factors to use in matrix 
        factorization model, some machine-learning libraries
        denote this as rank
        
    reg : float
        regularization term for item/user latent factors,
        since lambda is a keyword in python we use reg instead
    """

    def __init__(self, n_iters, n_factors, reg):
        self.reg = reg
        self.n_iters = n_iters
        self.n_factors = n_factors  
        
    def fit(self, train):
        """
        pass in training and testing at the same time to record
        model convergence, assuming both dataset is in the form
        of User x Item matrix with cells as ratings
        """
        self.n_user, self.n_item = train.shape
        self.user_factors = scsp.csr_matrix(np.random.uniform(size=(self.n_user, self.n_factors)))
        self.item_factors = scsp.csr_matrix(np.random.uniform(size=(self.n_item, self.n_factors)))
        
        # record the training and testing mse for every iteration
        # to show convergence later (usually, not worth it for production)
        # self.test_mse_record  = []
        self.train_mse_record = []   
        for i in range(self.n_iters):
            # print(i)
            self.user_factors = self._als_step(train, self.user_factors, self.item_factors)
            self.item_factors = self._als_step(train.T, self.item_factors, self.user_factors) 
            # predictions = self.predict()
            # test_mse = self.compute_mse(test, predictions)
            # train_mse = self.compute_mse(train, predictions)
            # self.test_mse_record.append(test_mse)
            # self.train_mse_record.append(train_mse)
        
        return self    
    
    def _als_step(self, ratings, solve_vecs, fixed_vecs):
        """
        when updating the user matrix,
        the item matrix is the fixed vector and vice versa
        """
        A = fixed_vecs.T.dot(fixed_vecs) + np.eye(self.n_factors) * self.reg
        b = ratings.dot(fixed_vecs)
        A_inv = np.linalg.inv(A)
        solve_vecs = b.dot(A_inv)
        return solve_vecs
    
    def predict(self):
        """predict ratings for every user and item"""
        pred = self.user_factors.dot(self.item_factors.T)
        return pred
    
    @staticmethod
    def compute_mse(y_true, y_pred):
        """ignore zero terms prior to comparing the mse"""
        mask = np.nonzero(y_true)
        mse = mean_squared_error(y_true[mask], y_pred[mask])
        return mse
    

als = ExplicitMF(n_iters = 100, n_factors = 40, reg = 0.01)
als.fit(M)

eval_df = interactions_test[['u', 'i', 'rating']]
eval_df.loc[:, 'rating_pred'] = np.nan

for (u, i) in zip(interactions_test['u'], interactions_test['i']):
    pred= als.user_factors[u].dot(als.item_factors[i].T).item()
    eval_df.loc[eval_df['u']==u, 'rating_pred'] = pred

print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))

sns.scatterplot(data=eval_df, x='rating', y='rating_pred')
