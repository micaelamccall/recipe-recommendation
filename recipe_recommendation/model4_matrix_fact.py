
import numpy as np
import seaborn as sns
from surprise import Dataset, Reader
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class NMF():
    
    def __init__(self, n_factors=15, n_epochs=50, biased=False, reg_pu=.06,
                 reg_qi=.06, lr=.005, random_state=None):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lr_bu = lr
        self.lr_bu = lr
        self.random_state = random_state


    def fit(self, trainset):

        self.trainset = trainset
        self.sgd(trainset)
        return self

    def sgd(self, trainset):

        # user and item factors
        pu = np.random.uniform(0, 0.5, size=(trainset.n_users, self.n_factors))
        qi = np.random.uniform(0, 0.5, size=(trainset.n_items, self.n_factors))

        n_factors = self.n_factors
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi
        global_mean = self.trainset.global_mean

        # auxiliary matrices used in optimization process
        user_num = np.zeros((trainset.n_users, n_factors))
        user_denom = np.zeros((trainset.n_users, n_factors))
        item_num = np.zeros((trainset.n_items, n_factors))
        item_denom = np.zeros((trainset.n_items, n_factors))

        if not self.biased:
            global_mean = 0

        self.losses = []

        for current_epoch in range(self.n_epochs):

            print("Processing epoch {}".format(current_epoch))

            user_num[:, :] = 0
            user_denom[:, :] = 0
            item_num[:, :] = 0
            item_denom[:, :] = 0

            epoch_loss = 0
            l = 0
            # Compute numerators and denominators for users and items factors
            for u, i, r in trainset.all_ratings():

                # compute current estimation and error
                dot = 0  # <q_i, p_u>
                for f in range(n_factors):
                    dot += qi[i, f] * pu[u, f]
                est = global_mean + dot
                err = r - est
                epoch_loss += err**2
                l += 1
                # compute numerators and denominators
                for f in range(n_factors):
                    user_num[u, f] += qi[i, f] * r
                    user_denom[u, f] += qi[i, f] * est
                    item_num[i, f] += pu[u, f] * r
                    item_denom[i, f] += pu[u, f] * est

            print(epoch_loss, l)
            epoch_loss = epoch_loss / l
            print(epoch_loss)
            self.losses.append(epoch_loss)
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


algo = NMF(n_factors=40, n_epochs=100)
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

eval_df.to_csv("results/MF.csv")

sns.scatterplot(data=eval_df, x='rating', y='rating_pred')


plt.plot(range(0,100,2), algo.losses[::2])
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("MSE across SGD iterations")
