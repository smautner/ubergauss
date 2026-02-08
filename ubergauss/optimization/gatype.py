from lmz import Map, Zip, Filter, Grouper, Range, Transpose, Flatten
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from hyperopt.pyll.stochastic import sample as hypersample
from pprint import pprint
from ubergauss.optimization import nutype as nt
from ubergauss.optimization import gansga as gans

"""
genetic optimization algorithm
"""
import random
from ubergauss.optimization import baseoptimizer


class nutype(baseoptimizer.base):
    def __init__(self, space, f, data, numsample=16, hyperband=[], **gaargs):
        super().__init__(space, f, data, numsample=numsample, hyperband=hyperband)
        self.seen = set()
        self.keyorder = list(self.params[0].keys())

        self.numsample_factor = 0.3
        self.maxold = 0.1
        self.mutation_rate = -1
        self.__dict__.update(gaargs)

        # UMAP visualization state
        self.umap_reducer = None

    # def hashconfig(self,p):
    #     return hash(tuple(p[k] for k in self.keyorder))
    # def register(self,p):
    #     for e in p:
    #         self.seen.add(self.hashconfig(e))

    def nuParams(self):
        select = int(self.numsample * self.numsample_factor)

        # Track which parents are selected from the last run
        selected_df = None

        if self.T == 0:
            selected_df = select_top(self.runs, select)
            pool, weights = df_to_params(selected_df, prin=False)

        if self.T == 1:
            selected_df = tournament_hist(self.runs, select, num_competitors=2)
            pool, weights = df_to_params(selected_df, prin=False)

        if self.T == 2:
            selected_df = new_pool_soft(self.runs, select, maxold=self.maxold)
            pool, weights = df_to_params(selected_df, prin=False)

        if self.T == 3:
            selected_df = tournament_plus(
                self.runs, select, num_competitors=2, old=self.maxold
            )
            pool, weights = df_to_params(selected_df, prin=False)

        # pool, weights = clusterpool(self.runs,self.space, select)
        # pool, weights = elitist_pool(self.runs, select)
        # pool, weights = tournament(self.runs,select)
        # pool, weights = toprando(self.runs,select)
        # pool, weights = expo_select(self.runs,select)
        weights = weights / np.sum(weights)

        # recombine
        new_params = []
        while len(new_params) < self.numsample:
            # x,y = np.random.choice(np.arange(len(pool)), size=2, replace=False, p=weights)
            x, y = np.random.choice(np.arange(len(pool)), size=2, replace=False)
            candidate = combine(pool[x], pool[y], self.space)
            # candidate = combine_aiming(pool[x],pool[y], weights[x] > weights[y], self.space)
            # candidate =  combine_dependant(pool[x],pool[y],self.paramgroups, self.space)
            # candidate =  combine_classic(pool[x],pool[y], self.space)
            new_params.append(candidate)
            # if self.hashconfig(candidate) not in self.seen:
            #     self.register([candidate])
            #     new_params.append(candidate)
            # else:
            #     print(self.seen)
            #     print(self.hashconfig(candidate))
            #     print(pool[x])
            #     print(pool[y])
            #     print(candidate)

        self.params = self.mutate(new_params)
        # self.params = new_params

        # take all the runs, vectorize them using GA
        # learn UMAP, plot last run using matplotlib (scores= viridis colors, log transformed as there are always a few very bad ones)
        # vectorize self.param and draw red dots
        # call here but write the code into gansga.py
        self.umap_reducer = gans.plot_umap_visualization(
            self.runs,
            self.params,
            self.space,
            self.umap_reducer,
            pool,
        )

        # wait for user to press enter
        input("Press Enter to continue...")

    def mutate(self, params):
        return [self.mutate_params(p) for p in params]

    def mutate_params(self, p):
        if self.mutation_rate < 0.0001:
            proba = 1 / (
                len(self.keyorder) + len(self.keyorder) * 0.2 * len(self.runs)
            )  # slowly decreasing mutation rate :)
            # proba =  1/(len(self.keyorder)+1) # slowly decreasing mutation rate :)
        for k in list(p.keys()):
            if (
                random.random() < proba
            ):  # + proba*isinstance(p[k], int): # double mutation rate for categoricals
                # Mutate by sampling a new value from the original search space
                p[k] = hypersample(self.space.hoSpace[k])
        return p


def df_to_params(dfdf, prin=False):
    # scores -= sorted.iloc[-5].score
    if prin:
        print(dfdf)
    scores = dfdf.score
    # dfdf = dfdf.drop(columns=['time', 'score','config_id'])
    dfdf = dfdf.drop(columns=["time", "score", "data_id"])
    pool = dfdf.to_dict(orient="records")
    weights = scores - min(scores) + 1  # np.argsort(np.array(scores)) + 3
    return pool, weights  # scores.tolist()


def avg_noise(a, b, key, space):
    typ = space.space[key][0]
    a, b = a[key], b[key]
    new = random.choice([a, b])
    new = np.mean([a, b])
    std = abs(a - b) * 0.5
    if typ == "int":
        std = max(std, 0.3)
    new = np.random.normal(new, std)
    # new = np.mean([a,b])
    low, high = space.space[key][1][:2]
    new = max(new, low)
    new = min(high, new)
    if typ == "int":
        new = int(new + 0.5)
    return new


def combine(a, b, space=None):
    new_params = {}
    for k in a.keys():
        val_a = a[k]
        val_b = b[k]
        typ = space.space[k][0]
        if typ == "cat":
            new_params[k] = random.choice([val_b, val_a])
            continue
        new_params[k] = avg_noise(a, b, k, space)
    return new_params


def combine_classic(a, b, space=None):
    new_params = {}
    keys = list(a.keys())
    num_keys = len(keys)
    crossover_point = random.randint(0, num_keys)
    for i, key in enumerate(keys):
        if i < crossover_point:
            new_params[key] = a[key]
        else:
            new_params[key] = b[key]
    return new_params


def new_pool_soft(runs, numselect, maxold=0.66):
    #
    # 1. select the best of the best
    n_combo = int(numselect * maxold)
    combo = pd.concat(runs)
    combo = combo.sort_values(by="score", ascending=False).head(n_combo)
    # 2. concatenate with newest, sort, remove duplicates and head
    final = pd.concat([combo, runs[-1]])
    final = final.sort_values(by="score", ascending=False)
    final = final.drop_duplicates().head(numselect)
    return final


# def new_and_clusters(runs, numselect, space):
#     combo = pd.concat(runs)
#     # combo = combo.sort_values(by='score', ascending=False).head(n_combo)
#     vectors = df_to_vec(combo,space)
#     # 3. select instances based on clusters and fitness
#     selected_indices = select_parents_by_cluster_fitness(vectors,
#                                                          combo['score'].tolist(),
#                                                          int(numselect*.66))
#     final = pd.concat([combo, runs[-1]])
#     final = final.sort_values(by='score', ascending=False)
#     final = final.drop_duplicates().head(numselect)
#     return df_to_params(final)


def clusterpool(runs, space, num_parents):
    # 1. make a copy of df and pop data_id time and score
    df = pd.concat(runs)
    vectors = gans.df_to_vec(df, space)
    # 3. select instances based on clusters and fitness
    scores = df["score"].tolist()
    selected_indices = select_parents_by_cluster_fitness(vectors, scores, num_parents)
    # 5. Filter the original df and call df_to_params
    selected_df = df.iloc[selected_indices].copy()
    return selected_df


def elitist_pool(runs, numselect):
    # SELECT THE BEST
    dfdf = pd.concat(runs)
    # dfdf = runs[-1]
    dfdf = dfdf[dfdf.score > 0]
    sorted = dfdf.sort_values(by="score", ascending=False)
    dfdf = sorted.head(numselect)
    # scores -= sorted.iloc[-5].score
    return dfdf


def expo_select(runs, num_select):
    df = pd.concat(runs)
    scores = df.score
    probabilities = scores**10 / np.sum(scores**10)
    selected_indices = np.random.choice(
        df.index, size=num_select, replace=False, p=probabilities
    )
    # print(selected_indices)
    dfdf = df.iloc[selected_indices]
    return dfdf


def toprando(runs, numselect):
    dfdf = pd.concat(runs)
    # len(runs)+1 * numselect *
    if len(runs) == 1:
        dfdf = dfdf.nlargest(numselect, "score")
    else:
        dfdf = dfdf.nlargest(numselect * 3, "score")
    return dfdf


def tournament(runs, numselect):
    dfdf = pd.concat(runs)
    dfdf = dfdf[dfdf.score > 0]
    bestof = len(dfdf) // numselect
    bestof = max(2, bestof)

    items = Zip(dfdf.score, Range(dfdf))

    def select():
        item = max(random.sample(items, bestof))
        items.remove(item)
        return item[1]

    indices = [select() for i in range(numselect)]
    dfdf = dfdf.iloc[indices]
    return dfdf


def tournament_plus(runs, num_select, num_competitors=2, old=0.1):
    # candidates:
    all_runs = pd.concat(runs)
    n_old = int(num_select * old)
    overall_best = all_runs.sort_values(by="score", ascending=False).iloc[0:n_old]
    df = pd.concat([overall_best, runs[-1]]).drop_duplicates().reset_index(drop=True)

    available = Range(len(df))
    selected_indices = set()
    # Perform tournament selection
    while len(selected_indices) < num_select:
        if num_competitors >= len(available):
            break
        competitors = df.iloc[available].sample(n=num_competitors, replace=False)
        winner_idx = competitors["score"].idxmax()
        selected_indices.add(winner_idx)
        # updata available
        available = [idx for idx in available if idx != winner_idx]

    return df.loc[list(selected_indices)].drop_duplicates()


def select_top(runs, num_select, num_competitors=2):
    # candidates:
    all_runs = pd.concat(runs)
    return all_runs.sort_values(by="score", ascending=False).iloc[0:num_select]


def tournament_hist(runs, num_select, num_competitors=2):
    # candidates:
    all_runs = pd.concat(runs)
    df = (
        all_runs.sort_values(by="score", ascending=False)
        .iloc[0 : int(len(all_runs) * 0.66)]
        .reset_index(drop=True)
    )

    available = Range(len(df))
    selected_indices = set()
    # Perform tournament selection
    while len(selected_indices) < num_select:
        if num_competitors >= len(available):
            break
        competitors = df.iloc[available].sample(n=num_competitors, replace=False)
        winner_idx = competitors["score"].idxmax()
        selected_indices.add(winner_idx)
        # updata available
        available = [idx for idx in available if idx != winner_idx]

    return df.loc[list(selected_indices)].drop_duplicates()


from sklearn.cluster import KMeans
from scipy.stats import rankdata


def select_parents_by_cluster_fitness(vectors, scores, num_parents, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(vectors)

    scores = np.array(scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    selected_ids = []

    cluster_means = []
    for c in range(num_clusters):
        idxs = np.where(labels == c)[0]
        cluster_score = scores[idxs].mean()
        cluster_means.append((c, cluster_score))

    # Rank the cluster means (higher score → higher rank)

    cluster_scores_only = [s for _, s in cluster_means]
    ranks = rankdata(cluster_scores_only, method="average")  # lowest=1, highest=n
    ranks = np.array([r > 1 for r in ranks])
    # ranks = ranks **2
    normalized_ranks = ranks / sum(
        ranks
    )  # (ranks - 1) / (len(ranks) - 1)  # scale to [0, 1]

    # Convert to proportions
    proportions = [(c, r) for (c, _), r in zip(cluster_means, normalized_ranks)]
    # print(f"{ proportions=}")

    for c, prop in proportions:
        # Determine how many parents to select from this cluster
        if prop < 0.0001:
            continue
        cluster_idxs = np.where(labels == c)[0]
        cluster_scores = scores[cluster_idxs]
        num_from_cluster = max(1, int(round(prop * num_parents)))

        # Select top individuals in cluster
        top_idxs = cluster_idxs[np.argsort(cluster_scores)[-num_from_cluster:]]
        print(f"{ len(top_idxs)=}")
        selected_ids.extend(top_idxs.tolist())

    # If rounding caused fewer than requested, fill in with next best
    if len(selected_ids) < num_parents:
        all_remaining = list(set(range(len(scores))) - set(selected_ids))
        top_up = sorted(all_remaining, key=lambda i: scores[i], reverse=True)[
            : (num_parents - len(selected_ids))
        ]
        selected_ids.extend(top_up)

    return selected_ids[:num_parents]


def combine_aiming(a, b, agb=False, space=None):
    new_params = {}
    for k in a.keys():
        val_a = a[k]
        val_b = b[k]
        typ = space.space[k][0]
        better, good = (val_a, val_b) if agb else (val_b, val_a)
        if typ == "cat":
            new_params[k] = random.choice([better, good])
            continue
        # new =  (better+good)/2
        new = np.random.normal(better, abs(better - good))
        low, high = space.space[k][1][:2]
        new = max(new, low)
        new = min(high, new)
        if typ == "int":
            new = int(new + 0.5)
        new_params[k] = new
    return new_params


def combine_dependant(a, b, paramgroups, space):
    new_params = a.copy()
    for keys_in_group in paramgroups:
        if random.random() < 0.5:  # 50% chance to inherit from 'b'
            for k in keys_in_group:
                if k in new_params and k in b:
                    new_params[k] = b[k]
        k = keys_in_group[0]
        if len(keys_in_group) == 1 and space.space[k][0] != "cat":
            new_params[k] = avg_noise(a, b, k, space)
        elif len(keys_in_group) == 2 and a[k] == b[k]:
            k2 = keys_in_group[1]
            new_params[k2] = avg_noise(a, b, k2, space)
    return new_params


from scipy.stats import gmean
import random


def test():
    """
    just run nutype on the benchmark problems... and report average score

    i think data_id is still in the data.. check if its added by the optimizer..    !!! TODO !!!
    maybe the optimizer needs to treat meta data seperately

    # in general we should do a grid search on the params... but then we need to expose everything.. difficult..
    """
    # from deap import benchmarks
    # functions = {
    #     "sphere": benchmarks.sphere,
    #     # "rastrigin": benchmarks.rastrigin,
    #     # "rosenbrock": benchmarks.rosenbrock,
    #     # "ackley": benchmarks.ackley,
    #     # "schwefel": benchmarks.schwefel,
    #     # "griewank": benchmarks.griewank,
    #     # "h1": benchmarks.h1  # Multi-modal function
    # }

    space = "\n".join([f"x{i} -5 5" for i in range(5)])

    def f(ads, **params):
        # res =  -func(list(params.values()))[0]

        res = list(params.values())
        # f = lambda a: ((a[0] + 1) * a[1]) ** 2 + random.random()
        # res = -sum(map(f, enumerate(res)))
        f = lambda x: x**2
        res = -sum(map(f, res[:2]))

        return res

    o = nutype(space, f, data=[[0]], numsample=128, T=0)
    o.opti()
    o.numsample = 32
    [o.opti() for i in range(10)]
    o.print()
    print(o.runs[-1])
    return o


def test_deap_functions():
    dim = 10  # 50-dimensional!
    x = np.random.uniform(-5, 5, dim)
    functions = {
        "sphere": benchmarks.sphere,
        "rastrigin": benchmarks.rastrigin,
        "rosenbrock": benchmarks.rosenbrock,
        "ackley": benchmarks.ackley,
        "schwefel": benchmarks.schwefel,
        "griewank": benchmarks.griewank,
        "h1": benchmarks.h1,  # Multi-modal function
    }
    for name, func in functions.items():
        result = func(x)[0]  # DEAP returns tuple
        print(f"{name:15} | dim={dim} | f(x) = {result:.6f}")
