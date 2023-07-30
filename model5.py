
import numpy as np
import seaborn as sns
from surprise import Dataset, Reader
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error



class NMF():
    
    def __init__(self, n_factors=15, n_epochs=50, biased=False, reg_pu=.06,
                 reg_qi=.06, reg_bu=.02, reg_bi=.02, lr_bu=.005, lr_bi=.005,
                 init_low=0, init_high=1, random_state=None, verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.init_low = init_low
        self.init_high = init_high
        self.random_state = random_state
        self.verbose = verbose

        if self.init_low < 0:
            raise ValueError('init_low should be greater than zero')


    def fit(self, trainset):

        self.trainset = trainset

        # AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        # user and item factors
        pu = np.random.uniform(self.init_low, self.init_high, size=(trainset.n_users, self.n_factors))
        qi = np.random.uniform(self.init_low, self.init_high, size=(trainset.n_items, self.n_factors))

        # user and item biases
        bu = np.zeros(trainset.n_users, dtype=np.double)
        bi = np.zeros(trainset.n_items, dtype=np.double)

        n_factors = self.n_factors
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi
        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        global_mean = self.trainset.global_mean

        # auxiliary matrices used in optimization process
        user_num = np.zeros((trainset.n_users, n_factors))
        user_denom = np.zeros((trainset.n_users, n_factors))
        item_num = np.zeros((trainset.n_items, n_factors))
        item_denom = np.zeros((trainset.n_items, n_factors))


        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):

            # if self.verbose:
            print("Processing epoch {}".format(current_epoch))

            # (re)initialize nums and denoms to zero
            # TODO: Use fill or memset??
            user_num[:, :] = 0
            user_denom[:, :] = 0
            item_num[:, :] = 0
            item_denom[:, :] = 0

            # Compute numerators and denominators for users and items factors
            for u, i, r in trainset.all_ratings():

                # compute current estimation and error
                dot = 0  # <q_i, p_u>
                for f in range(n_factors):
                    dot += qi[i, f] * pu[u, f]
                est = global_mean + bu[u] + bi[i] + dot
                err = r - est

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # compute numerators and denominators
                for f in range(n_factors):
                    user_num[u, f] += qi[i, f] * r
                    user_denom[u, f] += qi[i, f] * est
                    item_num[i, f] += pu[u, f] * r
                    item_denom[i, f] += pu[u, f] * est

            # Update user factors
            for u in trainset.all_users():
                n_ratings = len(trainset.ur[u])
                for f in range(n_factors):
                    if pu[u, f] != 0:  # Can happen if user only has 0 ratings
                        user_denom[u, f] += n_ratings * reg_pu * pu[u, f]
                        pu[u, f] *= user_num[u, f] / user_denom[u, f]

            # Update item factors
            for i in trainset.all_items():
                n_ratings = len(trainset.ir[i])
                for f in range(n_factors):
                    if qi[i, f] != 0:
                        item_denom[i, f] += n_ratings * reg_qi * qi[i, f]
                        qi[i, f] *= item_num[i, f] / item_denom[i, f]

        self.bu = np.asarray(bu)
        self.bi = np.asarray(bi)
        self.pu = np.asarray(pu)
        self.qi = np.asarray(qi)

    def predict(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if known_user and known_item:
            est = np.dot(self.qi[i], self.pu[u])
        else:
            est = None

        return est
    
interactions_train = pd.read_csv("data/interactions_train_mm.csv", index_col=0)
interactions_test = pd.read_csv("data/interactions_test_mm.csv", index_col=0)

interactions_train['rating'] += 1
interactions_test['rating'] += 1

reader = Reader(rating_scale=(1, 6))

traindata = Dataset.load_from_df(interactions_train[["u", "i", "rating"]], reader)
traindata = traindata.build_full_trainset()

testdata = Dataset.load_from_df(interactions_test[["u", "i", "rating"]], reader)
testdata = testdata.build_full_trainset().build_testset()


algo = NMF(n_factors=40,reg_pu=0.01, reg_qi=0.01, n_epochs=10)
algo.fit(traindata)


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
