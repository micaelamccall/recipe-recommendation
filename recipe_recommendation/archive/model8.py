
import numpy as np
import seaborn as sns
from surprise import Dataset, Reader
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.sparse as scsp


interactions_train_w_deets = pd.read_csv("data/interactions_train_w_deets.csv")[['u', 'i', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)
interactions_test_w_deets = pd.read_csv("data/interactions_test_w_deets.csv")[['u', 'i', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)
deets = pd.concat([interactions_train_w_deets['deets'], interactions_test_w_deets['deets']]).drop_duplicates().reset_index(drop=True)
deet_id_map = pd.DataFrame(deets).reset_index().rename(columns={'index':'d'})
interactions_train_w_deets = interactions_train_w_deets.merge(deet_id_map, how='left', on='deets')
interactions_test_w_deets = interactions_test_w_deets.merge(deet_id_map, how='left', on='deets')
i_new = interactions_train_w_deets[['i']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'i_new'})

id_map_train = interactions_train_w_deets[['i', 'd']].merge(i_new, how='left', on='i')
X = scsp.csr_matrix((np.ones((len(id_map_train))), (id_map_train['d'], id_map_train['i_new'])))
X.shape




interactions_train = pd.read_csv("data/interactions_train_mm.csv", index_col=0)
interactions_test = pd.read_csv("data/interactions_test_mm.csv", index_col=0)

interactions_train['rating'] += 1
interactions_test['rating'] += 1

reader = Reader(rating_scale=(1, 6))

trainset = Dataset.load_from_df(interactions_train[["u", "i", "rating"]], reader)
trainset = trainset.build_full_trainset()

n_factors = 40

n_deets = X.shape[0]
reg_pu = 0.01
reg_qi =0.01


        # user and item factors
pu = np.random.uniform(0, 1, size=(trainset.n_users, n_factors))
phi = np.random.uniform(0, 1, size=(n_deets, n_factors))
        


        # auxiliary matrices used in optimization process
user_num = np.zeros((trainset.n_users, n_factors))
user_denom = np.zeros((trainset.n_users, n_factors))
deet_num = np.zeros((n_deets, n_factors))
deet_denom = np.zeros((n_deets, n_factors))

for current_epoch in range(2):

    print("Processing epoch {}".format(current_epoch))

    # Compute numerators and denominators for users and items factors
    for u, i, r in trainset.all_ratings():
        print(u, i, r)

        # compute current estimation and error
        XPHI = X[:, i].T.dot(phi) # 1x40
        dot = 0  # <q_i, p_u>
        for f in range(n_factors):
            dot += XPHI[0, f] * pu[u, f]
        est = dot
        err = r - est
        print(est, r)
















testdata = Dataset.load_from_df(interactions_test[["u", "i", "rating"]], reader)
testdata = testdata.build_full_trainset().build_testset()


algo = NMF(n_factors=40, reg=0.01, n_epochs=2)
algo.fit(traindata, X)


eval_df = interactions_test[['u', 'i', 'rating']]
eval_df.loc[:, 'rating_pred'] = np.nan

for (u, i) in zip(interactions_test['u'], interactions_test['i']):
    pred= algo.predict(u, i)
    eval_df.loc[eval_df['u']==u, 'rating_pred'] = pred

eval_df.loc[eval_df['rating_pred'] > 6, 'rating_pred'] = 6
eval_df = eval_df[eval_df['rating_pred'].isna() == False].drop_duplicates()

print(np.sqrt(mean_squared_error(eval_df['rating'], eval_df['rating_pred'])))
print(mean_absolute_error(eval_df['rating'], eval_df['rating_pred']))

sns.scatterplot(data=eval_df, x='rating', y='rating_pred')
