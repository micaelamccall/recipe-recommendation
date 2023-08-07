
import numpy as np
import seaborn as sns
from surprise import Dataset, Reader
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.sparse as scsp


class NMF():
    
    def __init__(self, n_factors=15, n_epochs=50, biased=False, reg =.06,
                 lr=.005, random_state=None):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.reg_pu = reg
        self.reg_qi = reg
        self.lr_bu = lr
        self.lr_bu = lr
        self.random_state = random_state


    def fit(self, trainset, deetset):

        self.trainset = trainset
        self.X = deetset
        self.n_deets = deetset.shape[0]
        self.sgd(trainset, X)
        return self

    def sgd(self, trainset, X):

        # user and item factors
        pu = np.random.uniform(0, 1, size=(trainset.n_users, self.n_factors))
        phi = np.random.uniform(0, 1, size=(self.n_deets, self.n_factors))
        
        n_factors = self.n_factors
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi

        # auxiliary matrices used in optimization process
        user_num = np.zeros((trainset.n_users, n_factors))
        user_denom = np.zeros((trainset.n_users, n_factors))
        deet_num = np.zeros((self.n_deets, n_factors))
        deet_denom = np.zeros((self.n_deets, n_factors))


        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):

            print("Processing epoch {}".format(current_epoch))

            user_num[:, :] = 0
            user_denom[:, :] = 0
            deet_num[:, :] = 0
            deet_denom[:, :] = 0

            # Compute numerators and denominators for users and items factors
            for u, i, r in trainset.all_ratings():

                # compute current estimation and error
                XPHI = X[:, i].T.dot(phi) # 1x40
                dot = 0  # <q_i, p_u>
                for f in range(n_factors):
                    dot += XPHI[0, f] * pu[u, f]
                est = dot
                err = r - est
                # compute numerators and denominators
                for f in range(n_factors):
                    user_num[u, f] += XPHI[0, f] * r
                    user_denom[u, f] += XPHI[0, f] * est
                    deet_num[i, f] += (X[:, i] * pu[u, f])[0, f] * r
                    deet_denom[i, f] += (X[:, i] * pu[u, f])[0, f] * est

        #     # Update user factors
        #     for u in trainset.all_users():
        #         n_ratings = len(trainset.ur[u])
        #         for f in range(n_factors):
        #             if pu[u, f] != 0:  # Can happen if user only has 0 ratings
        #                 user_denom[u, f] += n_ratings * reg_pu * pu[u, f]
        #                 pu[u, f] *= user_num[u, f] / user_denom[u, f]

        #     # Update item factors
        #     for i in trainset.all_items():
        #         n_ratings = len(trainset.ir[i])
        #         for f in range(n_factors):
        #             if qi[i, f] != 0:
        #                 item_denom[i, f] += n_ratings * reg_qi * qi[i, f]
        #                 qi[i, f] *= item_num[i, f] / item_denom[i, f]

        # self.pu = np.asarray(pu)
        # self.qi = np.asarray(qi)

    def predict(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if known_user and known_item:
            est = np.dot(self.qi[i], self.pu[u])
        else:
            est = None

        return est
    

interactions_train_w_deets = pd.read_csv("data/interactions_train_w_deets.csv")[['u', 'i', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)
interactions_test_w_deets = pd.read_csv("data/interactions_test_w_deets.csv")[['u', 'i', 'deets', 'rating']].drop_duplicates().reset_index(drop=True)
deets = pd.concat([interactions_train_w_deets['deets'], interactions_test_w_deets['deets']]).drop_duplicates().reset_index(drop=True)
deet_id_map = pd.DataFrame(deets).reset_index().rename(columns={'index':'d'})
interactions_train_w_deets = interactions_train_w_deets.merge(deet_id_map, how='left', on='deets')
interactions_test_w_deets = interactions_test_w_deets.merge(deet_id_map, how='left', on='deets')
id_map_train = interactions_train_w_deets[['i', 'd']].drop_duplicates()
id_map_test = interactions_test_w_deets[['i', 'd']].drop_duplicates()
X = scsp.csr_matrix((np.ones((len(id_map_train))), (id_map_train['d'], id_map_train['i'])))
X.shape




interactions_train = pd.read_csv("data/interactions_train_mm.csv", index_col=0)
interactions_test = pd.read_csv("data/interactions_test_mm.csv", index_col=0)

interactions_train['rating'] += 1
interactions_test['rating'] += 1

reader = Reader(rating_scale=(1, 6))

traindata = Dataset.load_from_df(interactions_train[["u", "i", "rating"]], reader)
traindata = traindata.build_full_trainset()

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
