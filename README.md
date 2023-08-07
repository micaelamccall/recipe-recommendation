# Augmented recipe recommendation using flavor profile, ingredients, and cooking technique

*python | matrix algorithms | collaborative filtering | matrix factorization | content-based recommendation | hybrid recommendation*


## Background and Literature Review

As the amount of information on the internet has ballooned,
recommendation systems have become increasingly crucial to help users
find desired information without having to do extensive manual search.
The goal of a recommender system is to predict the rating a user would
give to a new item and to suggest to the user items for which the
predicted rating is high. Food and recipe recommendation is a domain in
which these systems are particularly relevant given the vast number of
online recipes. Recipe recommendation has gained traction in the
healthy-eating community as a way to suggest healthier meals or
ingredient substitutions \[1, 2, 3\]. It is also a relevant context for
group recommender system research; i.e., how to suggest recipes based on
the preferences of all members of a group \[4\].\
In addition to its many applications, basic recipe recommendation offers
fertile ground for researching the nuances of recommendation systems
because of the rich information about recipe contents (ingredients),
types of recipes (breakfast, dinner, desert, etc.), styles of cooking,
nutritional contents, etc. Some relevant questions in this area are: How
do we quantify a user's preference for specific ingredients given their
rating of a recipe and leverage this information to improve
recommendation? How do we incorporate preferences for particular types
of recipes? How do we build recommender systems under constraint, such
as only recommending recipes that conform to dietary restrictions?\
Researchers have tried to address some of these questions through
iteration on various approaches to algorithmic recommendation. Some of
the most common are:

**Collaborative filtering (CF)** is a method that uses the ratings of
many users over many items to identify similar users and predict the
rating a user would give to an item based on the ratings given by
similar users. The only data necessary for this approach is ratings
history on items \[5\].

**Content-based (CB)** is a method that uses information about items to
calculate similarity between new items and items a user has historically
rated to predict the rating a user would give to an item \[1\]. For
example, in the recipe/food domain, Freyne and Berkovsky (2010) broke
down ratings for a recipe into ratings for ingredients and then
reconstructed a prediction for a new recipe using a user's ratings for
the constituent ingredients.

**Knowledge-based (KB)** is a method that uses content knowledge about
the item as well as knowledge about users needs or a set of constraints
to recommend specific items and then includes an iterative process of
eliciting users' feedback. \[1, 7\]

**Hybrid approaches** are particularly used in recipe recommendation
because we have both plentiful user ratings data and about item content.
These approaches aim to combine the strengths of multiple previously
mentioned approaches. \[1, 7\]\
In my literature review, I noticed that many attempts to recommend
healthy foods used a KB approach; they didn't just reflect the user's
preferences, but also the nutritional needs of the particular user based
on their demographic information, e.g. in Alberg (2006). Attempts to
conform to dietary restrictions were treated similarly \[8\]. However, I
didn't find instances where rich recipe-based metadata was included in
recommender algorithms to try to improve the quality of the
recommendation, other than in Freyne and Berkovsky (2010).


# Project Goals

The aim of this project is to explore and compare the performance of CB,
CF, and hybrid recommender algorithms in the context of recipe
recommendation, while leveraging recipe flavor profiles and recipe
metadata (meal type, cooking technique, cooking time) in the
recommendation.\

# Data Sources

The first data set is a set of recipes from Food.com, made available on
as a Kaggle dataset
(<https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?resource=download&select=RAW_recipes.csv>).
The recipes data ncludes the name of the recipe, a description of the
recipe, recipe tags, the nutritional value, the steps in make the
recipe, and the ingredients.\
The second data set is a set of recipe interactions also from Food.com
and also made available through Kaggle
(<https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?resource=download&select=RAW_interactions.csv>).
This data set includes the rating and review given to a recipe by
various users (recipe id matches up to the recipe id in the recipes data
set above).\
The third data set provides the flavor molecules and associated flavor
profile for a given food (from
<https://cosylab.iiitd.edu.in/flavordb/>). It is accessed via API calls
for each ingredient in the recipes from the first datset.\
A final data set provides ingredient lemmatization, or in orther words
associating each possible ingredient with a \"base\" version of that
ingredient (e.g. all types of lettuce become \"lettuce\"). Also made
available through Kaggle
(<https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions/discussion/118716?resource=download&select=ingr_map.pkl>).

# Usage

Project scripts and Jupyter Notebook in the `recipe_recommendation` directory

- `recipe_recommendation.ipynb` is the notebook with all the script


<!-- # Methodology

## Data Preprocessing

For the recipe data set:

-   Ingredients will be extracted and associated with their \"base\"
    ingredient per the ingredient lemmatization map.

-   Tokenization of recipe tags.

-   Cooking techniques extracted from recipe steps.

\
For data fusion:

-   Each ingredient in each recipe queried for flavor profile.

-   Flavor profile of each ingredient aggregated across recipe.

-   Flavor profile associated with respective recipe.

\
Train-test split:

-   For interactions data set, data will be split into training,
    evaluation, and testing data sets. The interactions data will be the
    source of ground-truth that I will use to evaluate the
    recommendations. All/any recipes could be present in each of the
    train, eval, and test interactions data sets. -->

## Recommendation Algorithms

Each of the following algorithms was applied on the recipe interactions
data.\
**Baseline model**

First, built a model that randomly predicts ratings for a given user on
a given recipe by sampling from a uniform distribution.\
**Collaborative filtering (CF)**

**Nearest neighbors:** calculate the nearest neighbors of a new user
measured by cosine similarity:
$$Sim(u_i, u_k) := \frac{r_i * r_k}{||r_i||*||r_k||} = \frac{\sum_{j=1}^mr_{ij}r_{kj}}{\sqrt{\sum_{j=1}^mr_{ij}^2\sum_{j=1}^mr_{kj}^2}}$$
where $r_i$ and $r_k$ are ratings vectors for users $u_i$ and $u_k$.

Predict a user's rating on a new recipe $r_{ij}$ by weighted average
with bias avoided by by subtracting each user's average rating
$\tilde{r_k}$ from their rating of the recipe and adding in the target
user's average rating $\tilde{r_i}$:
$$r_{ij} = \tilde{r_i}+\frac{\sum_kSim(u_i, u_k)(r_{kj}-\tilde{r}_k)}{\text{num ratings}}$$

**Matrix Factorization:** aims to decompose the user's preferences for
into preferences for a set of latent factors. Matrix factorization can
be performed using Singular Value Decomposition (SVD):
$$M = U\Sigma V^T$$ 
By selecting the top $k$ singular values of matrix
$\Sigma$, we can reconstruct matrix $M$ with less dimensions but still
capturing much of the variability of the original matrix \[9\]. The
concept here, when applied over recipe ratings, would be to find the
dimensions of latent food preferences so as to avoid having to deal with
the high dimensionality of individual recipe ratings.

However, when factoring a sparse matrix, it's more efficient to use
Non-negative Matrix Factorization (NMF), which involves finding $P$ and
$Q$ such that the reconstructed user-item rating
$\hat{r}_{ui}= q_i^Tp_u$ is as close as possible to the true ${r}_{ui}$.
In order to find $P$ and $Q$, the Mean Squared Error is minimized:

$$min_{q,p} \sum_{(u, i) \in TR} (r_{ui} - q_i^Tp_u)^2 + \lambda(||q_i||^2+||p_u||^2)$$
where $p_u$ is the user vector, the $u$-th row of matrix $P$, and $q_i$
is the item vector, the $i$-th row of matrix $Q$, and $TR$ is the
training set \[9\]. I implemented this optimization by hand by
performing Gradient Decent according to the implementation in Luo et al.
(2014). On each update of the Gradient Decent, the entries of the $P$
and $Q$ matrices are updated as below:

$$p_{u,k} \leftarrow p_{u,k}\frac{\sum_{i \in TR}q_{k,i} r_{u,i}}{|I_u|\lambda p_{u,k} + \sum_{i \in TR} \hat r_{u,i}}$$

$$q_{k,i} \leftarrow q_{k,i}\frac{\sum_{i \in TR}p_{u,k} r_{u,i}}{|U_i|\lambda q_{k,i} + \sum_{i \in TR} \hat r_{u,i}}$$
where $I_u$ is the number of ratings for that user in the item set, and
$U_i$ is the number of rating for that item in the user set. A
prediction for a new user-recipe pair is simply the $\hat r_{ui}$ entry
in the reconstructed $\hat R = PQ^T$ matrix.\
**Content-based (CB)**

A rating for each user on each ingredient is calculated as the average
of the ratings each user gave to all recipes including that ingredient:
$$rat(u_i, ingr_j) = \frac{\sum_{l; ingr_j \in l}r_{il}}{l}$$ 
where
$r_{il}$ is the rating user $i$ gave to recipe $l$. This formula is then
applied over the flavor profile of each recipe, tags, and cooking
techniques, to create a comprehensive recipe-based data source for each
user. Predict a user's rating on a new recipe $r_{ij}$ by finding the
average rating across all the ingredients, flavors, and cooking
techniques in the new recipe:

$$r_{ij} = \frac{\sum_{l\in rec_j} rat(u_i, ingr_l)}{l}$$

**Hybrid**

**Content-augmented CF using cosine similarity**: attempts to generate
as many ratings as possible for a user on ingredients, flavors, and
techniques using ratings given by similar users, and then uses an
average of the ratings of the content of a recipe to predict a new
user-recipe rating.\
To be more specific, this involved three steps:

**1.** I used Equation 1 to find the nearest neighbors of a new user (as
in the CF approach).

**2.** Then, I used the following equation to predict a new user's
ratings on ingredients, flavors, and techniques that they hadn't already
rated:

$$rat(u_i, ingr_d) = \frac{\sum_k Sim(u_i, u_k)rat(u_k, ingr_d)}{\text{num ratings of }d}$$

**3.** Then I used Equation 8 to predict a user's rating on a new recipe
$r_{ij}$.

**Content-augmented matrix factorization**: takes the matrix
factorization approach to CF and augments the item data with content. I
used NMF for this approach, like the basic matrix factorization
approach. However, I incorporated recipe content information into this
factorization by further factoring the matrix $Q$ (shape: num features
by num recipes) as $X\Phi$, where $X$ is a matrix (shape: num recipes by
num ingredients) in which $X_{id}$ is a binary indicator if ingredient
$d$ is in recipe $i$. This results in the following NMF factorization of
matrix $R$: $$R = P\Phi^TX^T$$ My source for this approach is Forbes et
al. (2011). This updated NMF results in the following minimization of
MSE: $$min_{\phi,p} \sum_{(u,i) \in TR} (r_{ui} - p_u\Phi^Tx_i^T)^2$$

Rather than implementing the updates to each matrix on each iteration of
Gradient Descent by hand like I did for the basic matrix factorization,
I decided to use Pytorch to calculate and update the matrices according
to MSE loss. Once training is complete, a prediction for a new
user-recipe pair is simply the $\hat r_{ui}$ entry in the reconstructed
$\hat R = P\Phi^TX^T$ matrix.


# Evaluation and Final Results

I trained each model above using the training set described previously,
and then generated predictions using each user-recipe pair in the
testing set. I used Root Mean Square Error (RMSE), Mean Absolute Error
(MAE) and coverage (ability to generate predictions) \[6\] to evaluate
the performance of each approach.
|  Model                                  |  RMSE  |   MAE   |    Coverage |
|  ---------------------------------------| -------| --------| ----------|
|  Baseline                               |  2.5185|   2.04112|   1.0|
|  Vanilla CF                             |  0.9805|   0.5617|    1.0|
|  Matrix factorization CF                |  1.7859|   1.4959|    0.6837|
|  CB                                     |  1.3987|   1.0338|    0.3343|
|  Content-augmented CF                   |  1.0787|   0.8480|    0.9995|
|  Content-augmented matrix factorization |  4.1741|   3.8136|    1.1|


# References

1.  Trang Tran, T. N., Atas, M., Felfernig, A., & Stettinger, M. (2018).
    An overview of recommender systems in the healthy food domain.
    *Journal of Intelligent Information Systems*, 50 (pp. 501-526).

2.  Pecune, F., Callebert, L., & Marsella, S. (2020, September). A
    Recommender System for Healthy and Personalized Recipes
    Recommendations. In *HealthRecSys@ RecSys* (pp. 15-20).

3.  Van Pinxteren, Y., Geleijnse, G., & Kamsteeg, P. (2011, February).
    Deriving a recipe similarity measure for recommending healthful
    meals. In *Proceedings of the 16th international conference on
    Intelligent user interfaces* (pp. 105-114).

4.  Masthoff, J. (2011). Group recommender systems: Combining individual
    models. *Recommender systems handbook*, Springer (pp. 677--702).

5.  Ajitsaria, A. Build a Recommendation Enging with Collaborate
    Filtering. *RealPython.com*

6.  Freyne, J., & Berkovsky, S. (2010, February). Intelligent food
    planning: personalized recipe recommendation. In *Proceedings of the
    15th international conference on Intelligent user interfaces* (pp.
    321-324).

7.  Burke, R. (2002). Hybrid Recommender Systems: Survey and
    Experiments. *User Model User-Adap Inter* 12, (pp. 331--370).

8.  Aberg, J. (2006, January). Dealing with Malnutrition: A Meal
    Planning System for Elderly. In *AAAI spring symposium:
    argumentation for consumers of healthcare* (pp. 1-7).

9.  Luo, S. (2018, December). Introduction to Recommender System
    Approaches of Collaborative Filtering: Nearest Neighborhood and
    Matrix Factorization. *towardsdatascience.com*

10. Forbes, P., & Zhu, M. (2011, October). Content-boosted matrix
    factorization for recommender systems: experiments with recipe
    recommendation. In *Proceedings of the fifth ACM conference on
    Recommender systems* (pp. 261-264).

11. X. Luo, M. Zhou, Y. Xia and Q. Zhu. (May 2014). An Efficient
    Non-Negative Matrix-Factorization-Based Approach to Collaborative
    Filtering for Recommender Systems,\" in *IEEE Transactions on
    Industrial Informatics,* vol. 10, no. 2 (pp. 1273-1284)

