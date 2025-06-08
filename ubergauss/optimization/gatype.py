from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from hyperopt.pyll.stochastic import sample as hypersample
'''
for ints -> build a model cumsum(occcurance good / occurance bad) ,  then sample accoridngly
for floats -> gaussian sample from the top 50% of the scores
'''
import random
from ubergauss.optimization.nutype import base

class nutype(base):

    def __init__(self, space, f, data, numsample = 16,hyperband=[], floatavg =0):
        super().__init__( space, f, data, numsample = numsample, hyperband = hyperband )
        self.floatavg = floatavg
        self.seen = set()

    def hashconfig(self,p):
        return hash((p[k] for k in self.keyorder))
    def register(self,p):
        for e in p:
            self.seen.add(self.hashconfig(e))

    def nuParams(self):

        select  = int(self.numsample*.4)
        pool, weights = elitist_pool(self.runs, select)
        # pool, weights = self.loose_pool()
        weights = weights / np.sum(weights)
        # recombine
        def sample():
            x,y = np.random.choice(np.arange(len(pool)), size=2, replace=False, p=weights)
            assert x!=y
            return combine_aiming(pool[x],pool[y], weights[x] > weights[y], self.space)
            # return combine(pool[x],pool[y], self.space)
        new_params = [sample() for _ in range(self.numsample)]

        # mutate
        new_params = [ self.mutate(p,.05) for p in new_params]
        self.params = new_params


    def mutate(self, p, proba):
        for k in list(p.keys()):
            if random.random() < proba:# + proba*isinstance(p[k], int): # double mutation rate for categoricals
                # Mutate by sampling a new value from the original search space
                p[k] = hypersample(self.space.hoSpace[k])
        return p


# def combine( a, b, floatavg = True):
#     new_params = {}
#     for k in a.keys(): # assuming a and b have the same keys
#         val_a = a[k]
#         val_b = b[k]
#         # new_params[k] = random.choice([val_a, val_b])
#         if isinstance(val_a, int):
#             new_params[k] = random.choice([val_a, val_b])
#         elif isinstance(val_a, float):
#             mean_val = random.choice([val_a, val_b])
#             new_params[k] =  np.mean((val_a,val_b)) if floatavg else mean_val#np.random.normal(mean_val, std_val)
#     return new_params


def combine( a, b, space=None):
    new_params = {}
    for k in a.keys():
        val_a = a[k]
        val_b = b[k]
        typ = space.space[k][0]

        if typ == 'cat':
            new_params[k] = random.choice([val_b, val_a])
            continue
        # new = better + (better-good)/2
        new = random.choice([val_b, val_a])
        new = np.random.normal(new, abs(val_a - val_b))

        low, high = space.space[k][1][:2]
        new = max(new,low)
        new = min(high, new)
        if typ == 'int':
            new = int(new+.5)
        new_params[k] = new
    return new_params


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
        # new = better + (better-good)/2
        new = np.random.normal(better, abs(better-good))

        low, high = space.space[k][1][:2]
        new = max(new,low)
        new = min(high, new)
        if typ == 'int':
            new = int(new+.5)
        new_params[k] = new
    return new_params



def combine_dependant( a, b, floatavg = True, deps={},agb=False):
    new_params = {}
    for k in a.keys():
        if k in deps: # k has a dependency
            assert deps[k] in new_params # making sure the thing we depend on is there :)
        val_a = a[k]
        val_b = b[k]
        if isinstance(val_a, int):
            new_params[k] = random.choice([val_a, val_b])
            if k in deps:
                dep_var = deps[k]
                if a[dep_var] != b[dep_var]: # are the parrents different
                    # whiich parent are we following?
                    parentchoice_is_a = new_params[deps[k]] == a[deps[k]]
                    new_params[k] = val_a if parentchoice_is_a else val_b
        elif isinstance(val_a, float):
            mean_val = random.choice([val_a, val_b])
            new_params[k] =  np.mean((val_a,val_b)) if floatavg else mean_val#np.random.normal(mean_val, std_val)
            if k in deps:
                dep_var = deps[k]
                if a[dep_var] != b[dep_var]: # are the parrents different
                    parentchoice_is_a = new_params[deps[k]] == a[deps[k]]
                    new_params[k] = val_a if parentchoice_is_a else val_b
    return new_params


def loose_pool(runs, numold, numnew):
    # pick hald from the current pool
    new = runs[-1].sort_values(by='score', ascending=False).head(numnew)
    # pick 25 % old ones
    if len(runs) > 1:
        old = pd.concat(runs[:-1])
        old = old.sort_values(by='score', ascending=False).head(numold)
    # concat
    dfdf = pd.concat([old,new])
    scores =  dfdf.score.tolist()
    dfdf = dfdf.drop(columns=['time', 'score', 'datafield'])
    pool = dfdf.to_dict(orient='records')
    # add random instances
    # pool += [self.space.sample() for _ in range(self.numsample - len(pool))]
    weights= np.argsort(np.array(scores))+5
    return pool, weights




def elitist_pool(runs, numselect):
    # SELECT THE BEST
    dfdf = pd.concat(runs)
    # dfdf = runs[-1]
    dfdf = dfdf[dfdf.score > 0]
    sorted = dfdf.sort_values(by='score', ascending=False)
    dfdf = sorted.head(numselect)

    scores =  dfdf.score
    scores -= sorted.iloc[-1].score


    dfdf = dfdf.drop(columns=['time', 'score', 'datafield'])
    pool = dfdf.to_dict(orient='records')

    #weights= np.argsort(np.array(scores))+3

    return pool, scores.tolist()







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



# we want to write an evaluation function. it produced a function that has n integer args and m float args.
# when the correct integer is inputted (1) the score contribution mean is .5 otherwise 0, sd = 1
# for floats the target is a parabola with the highpoint between 0..10, maybe a bit flattened, again noise with sd=1 is added




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

