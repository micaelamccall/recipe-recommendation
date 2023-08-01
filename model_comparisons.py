import pandas as pd
import scipy.sparse as scsp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

interactions_test = pd.read_csv("data/interactions_test_mm.csv", index_col=0)[['user_id', 'recipe_id', 'u', 'i']].drop_duplicates()


model1 = pd.read_csv("results/model_1.csv", index_col=0)
model2 = pd.read_csv("results/model_2.csv", index_col=0)
model3 = pd.read_csv("results/model_3.csv", index_col=0)

model13 = model1.merge(model3, how='outer', on=['u', 'i', 'rating'])

model13['mean_pred_rating'] = model13[['rating_pred_x', 'rating_pred_y']].mean(axis=1)

print(np.sqrt(mean_squared_error(model13['rating'], model13['mean_pred_rating'])))
print(mean_absolute_error(model13['rating'], model13['mean_pred_rating']))

sns.scatterplot(data=model13, x='rating', y='mean_pred_rating')


model12 = model1.merge(model2, how='outer', on=['u', 'i', 'rating'])

model12['mean_pred_rating'] = model12[['rating_pred_x', 'rating_pred_y']].mean(axis=1)

print(np.sqrt(mean_squared_error(model12['rating'], model12['mean_pred_rating'])))
print(mean_absolute_error(model12['rating'], model12['mean_pred_rating']))

sns.scatterplot(data=model12, x='rating', y='mean_pred_rating')


comp = model1.merge(model3, how='outer', on=['u', 'i', 'rating'])
comp = comp.melt(id_vars=['u', 'i', 'rating'])

sns.boxplot(data=comp, x='rating', y='value', hue='variable')

plt.legend()
plt.show()
