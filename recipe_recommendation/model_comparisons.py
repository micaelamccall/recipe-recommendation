import pandas as pd
import scipy.sparse as scsp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(results, true_total, model_name):
    RMSE = np.sqrt(mean_squared_error(results['rating'], results['rating_pred']))
    MAE = mean_absolute_error(results['rating'], results['rating_pred'])
    coverage = len(results) / true_total

    print(f"{model_name} : RMSE: {RMSE} MAE: {MAE} Coverage: {coverage}")


def plot_results(results, model_name):
    sns.color_palette("crest")
    g = sns.boxplot(data=results, x='rating', y='rating_pred', palette='crest')
    g.set_title(f"{model_name} Results")
    plt.show()


if __name__ == "__main__":

    interactions_test = pd.read_csv("data/interactions_test_mm.csv", index_col=0)[['user_id', 'recipe_id', 'u', 'i']].drop_duplicates()

    true_total = len(interactions_test[['u', 'i']].drop_duplicates())


    BL = pd.read_csv("results/bl_model.csv", index_col=0)
    CF = pd.read_csv("results/CF.csv", index_col=0)
    CB = pd.read_csv("results/CB.csv", index_col=0)
    CA_CF = pd.read_csv("results/CA_CF.csv", index_col=0)
    MF = pd.read_csv("results/MF.csv", index_col=0)
    MF_hybrid = pd.read_csv("results/matrix_fact_hybrid.csv", index_col=0)


    calculate_metrics(BL, true_total, "BL")
    calculate_metrics(CF, true_total, "CF")
    calculate_metrics(CB, true_total, "CB")
    calculate_metrics(CA_CF, true_total, "Content-Augmented CF")
    calculate_metrics(MF, true_total, "Matrix Factorization")
    calculate_metrics(MF_hybrid, true_total, "Content-Augmented Matrix Factorization")

    plot_results(BL, "BL")
    plot_results(CF, "CF")
    plot_results(CB, "CB")
    plot_results(CA_CF, "Content-Augmented CF")
    plot_results(MF, "Matrix Factorization")
    plot_results(MF_hybrid, "Content-Augmented Matrix Factorization")


    r2_score(CF['rating'], CF['rating_pred'])
    r2_score(CB['rating'], CB['rating_pred'])
    r2_score(CA_CF['rating'], CA_CF['rating_pred'])
    r2_score(MF['rating'], MF['rating_pred'])
    r2_score(MF_hybrid['rating'], MF_hybrid['rating_pred'])
