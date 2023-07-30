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

comp = model1.merge(model3, how='outer', on=['u', 'i', 'rating'])
comp = comp.melt(id_vars=['u', 'i', 'rating'])

sns.boxplot(data=comp, x='rating', y='value', hue='variable')

plt.legend()
plt.show()
