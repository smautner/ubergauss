from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from hyperopt.pyll.stochastic import sample as hypersample
from pprint import pprint
'''
genetic optimizer
'''
import random
from ubergauss.optimization import baseoptimizer

class nutype(baseoptimizer.base):

    def __init__(self, space, f, data, numsample = 16,hyperband=[], floatavg =0):
        super().__init__( space, f, data, numsample = numsample, hyperband = hyperband )
        self.floatavg = floatavg
        self.seen = set()
        self.keyorder = list(self.params[0].keys())

    def hashconfig(self,p):
        return hash(tuple(p[k] for k in self.keyorder))
    def register(self,p):
        for e in p:
            self.seen.add(self.hashconfig(e))


    def nuParams(self):
        select  = int(self.numsample*.4)
        pool, weights = new_pool_soft(self.runs, select, maxold=.5)
        # pool, weights = tournament_usual(self.runs, select, 10)
        # pool, weights = tournament(self.runs, select, int(self.numsample*2 / select + .5))
        # pool, weights = elitist_pool(self.runs, select)
        # pool, weights = rando_pool(self.runs, select, self.space)
        # pool, weights = self.loose_pool()
        weights = weights / np.sum(weights)
        # recombine
        new_params= []
        while len(new_params) < self.numsample:
            # x,y = np.random.choice(np.arange(len(pool)), size=2, replace=False, p=weights)
            x,y = np.random.choice(np.arange(len(pool)), size=2, replace=False)
            # candidate = combine_aiming(pool[x],pool[y], weights[x] > weights[y], self.space)
            candidate =  combine(pool[x],pool[y], self.space)
            # candidate =  combine_dependant(pool[x],pool[y],self.paramgroups, self.space)
            # candidate =  combine_classic(pool[x],pool[y], self.space)
            if self.hashconfig(candidate) not in self.seen:
                self.register([candidate])
                new_params.append(candidate)

            # else:
            #     print(self.seen)
            #     print(self.hashconfig(candidate))
            #     print(pool[x])
            #     print(pool[y])
            #     print(candidate)


        new_params = [ self.mutate(p,1/(len(self.keyorder)+1)) for p in new_params]
        self.params = new_params

    def mutate(self, p, proba):
        for k in list(p.keys()):
            if random.random() < proba:# + proba*isinstance(p[k], int): # double mutation rate for categoricals
                # Mutate by sampling a new value from the original search space
                p[k] = hypersample(self.space.hoSpace[k])
        return p


def avg_noise(a,b,key,space):
    typ = space.space[key][0]
    a,b = a[key], b[key]
    new = np.mean([a,b])
    new = np.random.normal(new, abs(a-b)*.5)
    low, high = space.space[key][1][:2]
    new = max(new,low)
    new = min(high, new)
    if typ == 'int':
        new = int(new+.5)
    return new

def combine( a, b, space=None):
    new_params = {}
    for k in a.keys():
        val_a = a[k]
        val_b = b[k]
        typ = space.space[k][0]
        if typ == 'cat':
            new_params[k] = random.choice([val_b, val_a])
            continue
        new_params[k] = avg_noise(a,b,k,space)
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






def new_pool_soft(runs, numselect, maxold = .66):
    # hmm i dont even need to check the len runs :0 nice

    n_combo = int(numselect*maxold)
    combo = pd.concat(runs)
    combo = combo.sort_values(by='score', ascending=False).head(n_combo)

    final = pd.concat([combo, runs[-1]])
    final = final.sort_values(by='score', ascending=False)
    final = final.drop_duplicates().head(numselect)
    print(final)
    return df_to_params(final)



def df_to_params(dfdf):
    # scores -= sorted.iloc[-5].score
    scores =  dfdf.score
    dfdf = dfdf.drop(columns=['time', 'score', 'datafield'])
    pool = dfdf.to_dict(orient='records')
    weights= np.argsort(np.array(scores)) + 3
    return pool,weights # scores.tolist()


def elitist_pool(runs, numselect):
    # SELECT THE BEST
    dfdf = pd.concat(runs)
    # dfdf = runs[-1]
    dfdf = dfdf[dfdf.score > 0]
    sorted = dfdf.sort_values(by='score', ascending=False)
    dfdf = sorted.head(numselect)
    # scores -= sorted.iloc[-5].score
    print(dfdf)
    return df_to_params(dfdf)



import random
def tournament(runs, numselect, bestof = 8):
    dfdf = pd.concat(runs[-2:])
    dfdf = dfdf[dfdf.score > 0]
    items = Zip(dfdf.score, Range(dfdf))
    def select():
        random.shuffle(items)
        item = max(items[:8])
        items.remove(item)
        return item[1]
    indices = [select() for i in range(numselect)]
    dfdf = dfdf.iloc[indices]
    print(dfdf)
    return df_to_params(dfdf)

def tournament_usual(runs, numselect, bestof = 8):
    dfdf = pd.concat(runs)
    dfdf = dfdf[dfdf.score > 0]
    items = Zip(dfdf.score, Range(dfdf))
    def select():
        item = max(random.sample(items, bestof))
        items.remove(item)
        return item[1]
    indices = [select() for i in range(numselect)]
    dfdf = dfdf.iloc[indices]
    print(dfdf)
    return df_to_params(dfdf)



def fix_dataindex(df):
    # Identify columns that define a unique parameter combination (all except 'score' and 'datafield')
    param_cols = [col for col in df.columns if col not in ['score', 'datafield' ,'time']]

    # Calculate the average score for each parameter combination across all datafields
    avg_scores = df.groupby(param_cols)['score'].mean().reset_index()

    # Rename the calculated average score column to 'score'
    avg_scores = avg_scores.rename(columns={'score': 'average_score'})

    # Filter the original DataFrame to keep only rows where datafield is 0
    df_filtered = df[df['datafield'] == 0].copy()

    # Merge the calculated average scores back into the filtered DataFrame
    # The 'score' column in df_filtered will be replaced by the 'average_score' from avg_scores
    df_fixed = pd.merge(df_filtered.drop(columns='score'), avg_scores, on=param_cols, how='left')

    # Rename the 'average_score' column back to 'score'
    df_fixed = df_fixed.rename(columns={'average_score': 'score'})


    # Return the DataFrame with scores fixed and filtered to datafield == 0
    return df_fixed

def plot_params_with_hist(params, df):
    params = pd.DataFrame(params)
    for col in params.columns:
        if col == "score":
            continue  # Skip the score column itself

        fig, ax1 = plt.subplots(figsize=(8, 4))

        # Lineplot: param vs score
        sns.scatterplot(x=col, y="score", data=df, ax=ax1, color='blue', label='Score')
        ax1.set_ylabel("Score", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Histogram: distribution of values in df
        ax2 = ax1.twinx()
        sns.histplot(params[col], ax=ax2, color='gray', alpha=0.3, bins=20, label='Distribution')
        ax2.set_ylabel("Frequency", color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Titles and layout
        plt.title(f"{col}")
        fig.tight_layout()
        plt.show()







def combine_aiming( a, b, agb=False, space=None):
    new_params = {}
    for k in a.keys():
        val_a = a[k]
        val_b = b[k]
        typ = space.space[k][0]
        better, good = (val_a, val_b) if agb else (val_b,val_a)
        if typ == 'cat':
            new_params[k] = random.choice([better,good])
            continue
        #new =  (better+good)/2
        new = np.random.normal(better, abs(better-good))
        low, high = space.space[k][1][:2]
        new = max(new,low)
        new = min(high, new)
        if typ == 'int':
            new = int(new+.5)
        new_params[k] = new
    return new_params


def combine_dependant(a, b, paramgroups, space):
    new_params = a.copy()
    for  keys_in_group in paramgroups:
        if random.random() < 0.5: # 50% chance to inherit from 'b'
            for k in keys_in_group:
                if k in new_params and k in b:
                     new_params[k] = b[k]
        k=keys_in_group[0]
        if len(keys_in_group) == 1 and space.space[k][0] != 'cat':
            new_params[k] = avg_noise(a,b,k,space)
        elif len(keys_in_group) == 2 and a[k]==b[k]:
            k2 = keys_in_group[1]
            new_params[k2] = avg_noise(a,b,k2,space)
    return new_params


def create_evaluation_function(n_int, m_float):
    """
    Creates a synthetic evaluation function for optimization testing.

    The function evaluates a dictionary of parameters.
    Integer parameters keyed 'int_i' contribute 0.5 to the score if their
    value is exactly 1 (plus noise). Otherwise, they contribute 0 (plus noise).
    Float parameters keyed 'float_j' contribute a score based on a Gaussian-like
    function peaked at 5 (range 0-10 typically expected), plus noise.
    Noise (std=1) is added independently to each parameter's contribution.

    Args:
        n_int (int): The number of integer parameters (int_0, int_1, ...).
        m_float (int): The number of float parameters (float_0, float_1, ...).

    Returns:
        function: An evaluation function f(params, datafield=None) suitable
                  for use with optimization libraries like ubergauss/hyperopt.
    """
    def evaluation_function(params, datafield=None):
        """
        Synthetic evaluation function based on parameter values.

        Args:
            params (dict): Dictionary of parameters to evaluate.
                           Expected keys are 'int_0'...'int_{n_int-1}' and
                           'float_0'...'float_{m_float-1}'.
            datafield: Placeholder for potential data context (ignored here).

        Returns:
            float: The calculated score for the given parameters.
        """
        total_score = 0.0
        noise_std = 1.0

        # Contributions from integer parameters
        for i in range(n_int):
            key = f'int_{i}'
            value = params.get(key)

            # Check if the parameter exists and is an integer
            if isinstance(value, int):
                if value == 1:
                    # Correct integer value
                    mean_contrib = 0.5
                else:
                    # Incorrect integer value
                    mean_contrib = 0.0
            else:
                # Parameter missing or not an integer type
                mean_contrib = 0.0 # Treat as incorrect value

            total_score += mean_contrib + np.random.normal(0, noise_std)

        # Contributions from float parameters
        for i in range(m_float):
            key = f'float_{i}'
            value = params.get(key)

            # Check if the parameter exists and is numeric
            if isinstance(value, (float, int)):
                # Target shape: Gaussian peak at 5.0, scaled to contribute max 0.5
                # 0.1 controls the width of the peak (smaller value -> wider peak)
                target_value = 5.0
                peak_contribution = 0.5
                width_factor = 0.1 # Controls how steep the parabola/gaussian is

                score_contribution = peak_contribution * np.exp(-width_factor * (value - target_value)**2)
            else:
                # Parameter missing or not a numeric type
                score_contribution = 0.0 # Treat as poor value

            total_score += score_contribution + np.random.normal(0, noise_std)

        return total_score

    return evaluation_function

