from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from ubergauss import hyperopt as ho
from ubergauss import optimization as op
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pprint import pprint
from hyperopt.pyll.stochastic import sample as hypersample
'''
for ints -> build a model cumsum(occcurance good / occurance bad) ,  then sample accoridngly
for floats -> gaussian sample from the top 50% of the scores
'''
import random

class nutype:

    def __init__(self, space, f, data, numsample = 16):
        self.f = f
        self.data = data
        self.numsample = numsample
        self.space  = ho.spaceship(space)
        self.params =  [self.space.sample() for x in range(self.numsample) ]
        self.runs = []


    def opti(self):

        df = op.gridsearch(self.f, data_list = self.data,tasks = self.params)
        df = fix_dataindex(df)
        df = df.fillna(0)
        df = df.sort_values(by='score', ascending=True)
        self.runs.append(df)

        self.nuParams()

    def nuParams(self):
        pool, weights = elitist_pool(self.runs,self.numsample // 2)
        # pool, weights = self.loose_pool()
        weights = weights / np.sum(weights)

        # recombine
        def sample():
            x,y = np.random.choice(Range(len(pool)), size=2, replace=False, p=weights)
            assert x!=y
            return combine(pool[x],pool[y])
        new_params = [sample() for _ in range(self.numsample)]

        # mutate
        new_params = self.mutate(new_params,.05)

        self.params = new_params


    def mutate(self, params, proba):
        for p in params:
            for k in list(p.keys()): # Create a list copy to allow modification during iteration
                 if random.random() < proba:
                     # Mutate by sampling a new value from the original search space
                     p[k] = hypersample(self.space.space[k])
        return params



    def print(self):
        dfdf = pd.concat(self.runs)
        # print the params with the highest score
        best_run = dfdf.sort_values(by='score', ascending=False).iloc[0]
        pprint(best_run.drop(['score', 'time', 'datafield']).to_dict())

        mi = min(dfdf.score[:10])
        plotscores = [ s if s > mi else mi for s in dfdf.score]
        plt.plot(plotscores)
        plt.show()
        plot_params_with_hist(self.params,dfdf)



def combine( a, b):
    new_params = {}
    for k in a.keys(): # assuming a and b have the same keys
        val_a = a[k]
        val_b = b[k]
        # new_params[k] = random.choice([val_a, val_b])
        if isinstance(val_a, int):
            new_params[k] = random.choice([val_a, val_b])
        elif isinstance(val_a, float):
            mean_val = random.choice([val_a, val_b])
            std_val = abs(val_a - val_b) #/ 3.0
            if std_val < 1e-6: # Use a small epsilon
                std_val = 1e-6
            new_params[k] = np.mean((val_a,val_b))#np.random.normal(mean_val, std_val)
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
    dfdf = pd.concat(runs)
    dfdf = dfdf.sort_values(by='score', ascending=False).head(numselect)
    scores =  dfdf.score.tolist()
    dfdf = dfdf.drop(columns=['time', 'score', 'datafield'])
    pool = dfdf.to_dict(orient='records')
    weights= np.argsort(np.array(scores))+10
    return pool, weights








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


