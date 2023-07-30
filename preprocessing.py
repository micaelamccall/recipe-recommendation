import urllib.request
import json
import pandas as pd
from ast import literal_eval
import spacy
import scipy.stats as spst
import seaborn as sns
import matplotlib.pyplot as plt
nlp = spacy.load("en_core_web_sm")

ingr_map = pd.read_pickle("food.com_recipes/ingr_map.pkl")

recipes = pd.read_csv("food.com_recipes/RAW_recipes.csv")

pp_recipes = pd.read_csv("food.com_recipes/PP_recipes.csv")


def nounify_ingredient_list(ingredient_list):
    nounified_list = []
    for ing in ingredient_list:
        nouns = " ".join([t.text for t in nlp(ing) if t.pos_ == 'NOUN' or t.pos_ == "PROPN"]) 
        nounified_list.append(nouns)
    return nounified_list


def nounify_ingredient(ingredient_str):
    if len(ingredient_str.split(" ")) == 1:
        return ingredient_str
    else:
        return " ".join([t.text for t in nlp(ingredient_str) if t.pos_ == 'NOUN' or t.pos_ == "PROPN"])

def build_ingredient_dict(ingr_map):
    # filter by if the count in the data is > 10 
    filt_ingr = ingr_map[ingr_map['count'] > 10][['id', 'replaced']].drop_duplicates()
    # keep only nouns
    filt_ingr['ingredients'] = filt_ingr['replaced'].apply(nounify_ingredient)
    ingr_dict = dict(zip(filt_ingr['id'], filt_ingr['ingredients']))
    return ingr_dict


ingr_dict = build_ingredient_dict(ingr_map)

with open('data/ingr_dict.txt', 'w') as f:
    f.write(json.dumps(ingr_dict))


def find_flavors(ingredient):
    search_term = ingredient.replace(" ", "&")
    print(search_term)

    flavors = []
    
    bu = f"https://cosylab.iiitd.edu.in/flavordb/entities?entity={search_term}&category="
    with urllib.request.urlopen(bu) as url:
        b = json.loads(json.loads(url.read().decode()))

        # find the result that has the ingredient
        b = list(filter(lambda x: x['entity_alias_readable'].lower() in ingredient, b))
        # if nothing is returned, end func
        if len(b) == 0:
            return []
        # if none of them have the exact ingredient, the list will still be longer than 1; just return the first ingredient
        # if type(b) == list:
        #     b = b[0]
        print(f"found ingredient: {b[0]['entity_alias_readable']}")
    flavdb_id = b[0]['entity_id']
    bu = f"https://cosylab.iiitd.edu.in/flavordb/entities_json?id={flavdb_id}"
    with urllib.request.urlopen(bu) as url:
        a = json.loads(url.read().decode())
    for mol in a['molecules']:
        flavors.extend(mol['fooddb_flavor_profile'].split("@"))

    return list(set(flavors))[1:]



ing_flav_dict = {}

j = 0
for i, ing in ingr_dict.items():
    # j += 1
    # if j > 2500:
        # print(f"Iteration {j}")
        if len(ing) > 0:
            ing_flav_dict[ing] = find_flavors(ing)


with open('data/ing_flav_dict.txt', 'w') as f:
    f.write(json.dumps(ing_flav_dict))


with open('data/ing_flav_dict.txt', 'r') as f:
    ing_flav_dict = json.load(f)


#### Create id - flav dict
flavs = []

for _, flav in ing_flav_dict.items():
    flavs.extend(flav)
    

flavs = pd.Series(flavs).drop_duplicates()

flavs = flavs.reset_index(drop=True).reset_index().rename(columns={'index':'flavor_id', '0': 'flavor'})

flav_dict = {k:v for k, v in zip(flavs['flavor_id'], flavs[0])}

with open('data/flav_dict.txt', 'w') as f:
    f.write(json.dumps(flav_dict))

# for ing, flav in ing_flav_dict2.items():
#     ing_flav_dict[ing] = flav 



# ingredient = "tomato sauce"
# search_term = ingredient.replace(" ", "&")
# bu = f"https://cosylab.iiitd.edu.in/flavordb/entities?entity={search_term}&category="
# with urllib.request.urlopen(bu) as url:
#     b = json.loads(json.loads(url.read().decode()))


pp_ingr = pp_recipes[['id', 'ingredient_ids']]
pp_ingr['ingredient_ids'] = pp_ingr['ingredient_ids'].apply(json.loads)

from wordcloud import WordCloud

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(" ".join(ing_flav_dict['lettuce']))
 
# plot the WordCloud image                      
plt.figure(figsize = (16, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()

def ingr_match_and_filter(ingr_ids):
    try:
        ingr_list = [ingr_dict[i] for i in ingr_ids]
    except KeyError:
        ingr_list = None
    return ingr_list

pp_ingr['ingredients'] = pp_ingr['ingredient_ids'].apply(ingr_match_and_filter)

pp_ingr = pp_ingr[pp_ingr['ingredients'].isna() == False]

def flav_match(ingredients):
    # print(ingredients)
    flav_list = []
    for i in ingredients:
        try:
            flav_list_i = ing_flav_dict[i]
            # print(i, flav_list_i)
        except KeyError:
            continue
        if len(flav_list_i) >0:
            flav_list.extend(flav_list_i)
    flav_list = list(set(flav_list))
    return flav_list


pp_ingr['flavors'] = pp_ingr['ingredients'].apply(flav_match)

# pp_ingr['nouns'] = pp_ingr['ingredients'].apply(nounify_ingredient_list)

pp_ingr.to_csv("pp_ingr.csv")

# pp_ingr['nouns'] = pp_ingr['nouns'].apply(lambda x: [i for i in x if len(i) > 0])


# tags = []
# for l in recipes['tags']:
#     tags.extend(eval(l))
# tags = list(set(tags))


with open("data/techniques.txt", 'r') as f:
    techniques = [line.strip() for line in f]


pp_recipes = pp_recipes.rename(columns={'techniques':'technique_onehot'})
pp_recipes['techniques'] = pp_recipes['technique_onehot'].apply(lambda x: [ing for (ing, onehot) in zip(techniques, json.loads(x)) if onehot ==1])
pp_recipes.to_csv("data/pp_recipes.csv")

interactions_train = pd.read_csv("food.com_recipes/interactions_train.csv")
interactions_test = pd.read_csv("food.com_recipes/interactions_test.csv")

interactions = pd.concat([interactions_train, interactions_test])



agg = interactions.groupby(["recipe_id", 'i']).agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count'))

# agg.to_csv("data/agg.csv")

def plot_ratings_dist(agg):
    fig = plt.figure(figsize=(12, 5))
    fig_rows = 1
    fig_columns = 2
    fig.add_subplot(fig_rows, fig_columns, 1)
    plt.hist(agg['mean_rating'], bins=10, ec='white', fc='purple')
    plt.title('Dist of mean rating for recipes')
    fig.add_subplot(fig_rows, fig_columns, 2)
    plt.hist(agg['number_of_ratings'], bins=1000, ec='white', fc='purple')
    plt.xlim(0,50)
    plt.title('Dist of number of ratings for recipes')
    plt.show()

plot_ratings_dist(agg)


select_one_of_each = agg[agg['number_of_ratings'] > 10].reset_index()['recipe_id']

interactions[interactions['recipe_id'].isin(select_one_of_each)].shape

interactions_test_new = pd.DataFrame(columns=interactions.columns)

for id in select_one_of_each:
    interactions_test_new = pd.concat([interactions_test_new, interactions[interactions['recipe_id'] == id].sample(n=1)])

interactions_train_new = interactions.loc[~interactions.index.isin(interactions_test_new.index)]

interactions_test_new = interactions_test_new[interactions_test_new['user_id'].isin(interactions_train_new['user_id'])]

interactions_train_new.to_csv("data/interactions_train_mm.csv")
interactions_test_new.to_csv("data/interactions_test_mm.csv")

######ADDING DEETS ####

ingr = pd.read_csv("data/pp_ingr.csv", index_col=0)
ingr = ingr.rename(columns={'id':'recipe_id'})
interactions_train = pd.read_csv("data/interactions_train_mm.csv")[['user_id', 'recipe_id', 'rating', 'u', 'i']]
interactions_test = pd.read_csv("data/interactions_test_mm.csv")[['user_id', 'recipe_id', 'rating', 'u', 'i']]
pp_recipes = pd.read_csv("data/pp_recipes.csv")
pp_techniques = pp_recipes[['id', 'techniques']]
pp_techniques = pp_techniques.rename(columns={'id': 'recipe_id'})

interactions_train['rating'] += 1
interactions_test['rating'] += 1

def literal_return(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val

def add_deets_to_recipe(interactions, techniques, ingredients):
    interactions_w_deets = interactions.merge(techniques, how='left', on='recipe_id').merge(ingredients, how='left', on='recipe_id').drop(columns=['ingredient_ids'])
    interactions_w_deets['ingredients'] = interactions_w_deets['ingredients'].apply(literal_return)
    interactions_w_deets['techniques'] = interactions_w_deets['techniques'].apply(literal_return)
    interactions_w_deets['flavors'] = interactions_w_deets['flavors'].apply(literal_return)
    interactions_w_deets['deets'] = interactions_w_deets['ingredients'] + interactions_w_deets['techniques'] + interactions_w_deets['flavors']
    interactions_w_deets = interactions_w_deets.explode('deets')
    return interactions_w_deets
    
# Add details to training data
interactions_train_w_deets = add_deets_to_recipe(interactions_train, pp_techniques, ingr)
interactions_train_w_deets = interactions_train_w_deets[['user_id', 'recipe_id', 'deets', 'rating', 'u', 'i']].drop_duplicates().reset_index(drop=True)
interactions_train_w_deets.to_csv("data/interactions_train_w_deets.csv")

interactions_test_w_deets = add_deets_to_recipe(interactions_test, pp_techniques, ingr)
interactions_test_w_deets = interactions_test_w_deets[['user_id', 'recipe_id', 'deets', 'rating', 'u', 'i']].drop_duplicates().reset_index(drop=True)
interactions_test_w_deets.to_csv("data/interactions_test_w_deets.csv")

###### PLOTS ###########

interactions_test_w_deets = pd.read_csv("data/interactions_test_w_deets.csv", index_col=0)
interactions_test_w_deets[interactions_test_w_deets['i'] == 120]

import networkx as nx

G = nx.from_pandas_edgelist(interactions_test_w_deets[interactions_test_w_deets['i'] == 120][:150], 'i', 'deets')
nx.draw(G, with_labels=True, pos=nx.spring_layout(G, k=.3), font_size=8, edge_color='lightgray', node_color='lightblue', node_size=0, edgecolors='lightgray')
plt.show()

v = interactions_test_w_deets.groupby(["i"]).count()

pp_ingr['ingredients'][0]


def literal_return(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val
pp_ingr['flavors'] = pp_ingr['flavors'].apply(literal_return)

flav_explode = pp_ingr.explode('flavors')

flav_explode = flav_explode.groupby("flavors").agg(mentions = ('id', 'count')).reset_index()


sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(6, 6))
sns.set_color_codes("pastel")
sns.barplot(x="mentions", y="flavors", data=flav_explode.sort_values(by='mentions').sample(50),
            label="Total mentions")
plt.yticks(fontsize=8)
# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")


#### MAKING SMALLER DS #####



interactions_small = interactions.sample(frac=.1)[['date', 'rating', 'user_id', 'recipe_id']]

user_id_map = interactions_small[['user_id']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'u'})
recipe_id_map = interactions_small[['recipe_id']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'i'})

interactions_small = interactions_small.merge(user_id_map, how='left', on='user_id')
interactions_small = interactions_small.merge(recipe_id_map, how='left', on='recipe_id')


agg = interactions_small.groupby(["recipe_id", 'i']).agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count'))

def plot_ratings_dist(agg):
    fig = plt.figure(figsize=(12, 5))
    fig_rows = 1
    fig_columns = 2
    fig.add_subplot(fig_rows, fig_columns, 1)
    plt.hist(agg['mean_rating'], bins=10, ec='white', fc='purple')
    plt.title('Dist of mean rating for recipes')
    fig.add_subplot(fig_rows, fig_columns, 2)
    plt.hist(agg['number_of_ratings'], bins=1000, ec='white', fc='purple')
    plt.xlim(0,50)
    plt.title('Dist of number of ratings for recipes')
    plt.show()

plot_ratings_dist(agg)


select_one_of_each = agg[agg['number_of_ratings'] > 10].reset_index()['i']

interactions_small[interactions_small['i'].isin(select_one_of_each)].shape

interactions_test_small_new = pd.DataFrame(columns=interactions.columns)

for id in select_one_of_each:
    interactions_test_small_new = pd.concat([interactions_test_small_new, interactions_small[interactions_small['i'] == id].sample(n=1)])

interactions_train_small_new = interactions_small.loc[~interactions_small.index.isin(interactions_test_small_new.index)]

interactions_test_small_new = interactions_test_small_new[interactions_test_small_new['u'].isin(interactions_train_small_new['i'])]

interactions_train_small_new.to_csv("data/interactions_train_small_mm.csv")
interactions_test_small_new.to_csv("data/interactions_test_small_mm.csv")