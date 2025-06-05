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
we want to do 1d gridsearch.

do numsample random searches to initialize.

group the parameters, non-dependant ones get their own group, depentand ones go together.
there is no complicated nesting. just dependee->master, they dont show up multiple times in the list

then do random search on each group in turn, until all groups have been optimized once.
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
        self.dependencies = self.space.dependencies
        self.grp= []
        self.exhausted = set()
        self.paramgroups = self.getgroups()

    def getgroups(self):
        keys = {k:[k] for k in self.params[0].keys()}
        for slave, master in self.dependencies.items():
            keys.pop(slave)
            keys[master].append(slave)
        r =  list(keys.values())
        r = [ [rrr for rrr in rr if rrr not in self.exhausted] for rr in r   ]
        return r

    def opti(self):
        df = op.gridsearch(self.f, data_list = self.data,tasks = self.params)
        df = fix_dataindex(df)
        df = df.fillna(0)
        df = df.sort_values(by='score', ascending=True)

        self.runs.append(df)
        self.currentparams = df.iloc[-1].drop(['score', 'time', 'datafield']).to_dict()
        self.updateSpace()
        self.nuParams()
        return self



    def updateSpace(self):

        if len(self.grp) == 1:
            e = self.grp[0]
            avg_scores_by_param_value = self.runs[-1].groupby(e)['score'].mean().reset_index()
            avg_scores_by_param_value = avg_scores_by_param_value.sort_values(by='score', ascending=False)
            param_type, param_config = self.space.space[e]
            top = avg_scores_by_param_value.head(int(self.numsample * .5))[e]
            # print(f"optimizing le spave{e=}{min(top) =}")
            self.space.space[e][1][0] = min(top)
            self.space.space[e][1][1] = max(top)

        if len(self.grp) == 2:
            for a in self.grp:
                if self.space.space[a][0] == 'cat':
                    self.exhausted.add(a)


    def nuParams(self):
        # take params from self.paramgroupo, if empty reinitialize
        if not self.paramgroups: self.paramgroups = self.getgroups()
        grp = self.paramgroups.pop()
        grp = {p:list() for p in grp}

        # is one of the things is a categorical, select all categories
        # if len-grp is 2, select numsample/2 for this parameter (if not categorical)
        # else just select  numsamples in linspace
        for k in grp:

            if self.space.space[k][0] == 'cat':
                grp[k] = self.space.space[k][1] # all the things
                continue
            # int or float
            mynumspace = self.numsample // len(grp)
            start, end = self.space.space[k][1][:2]
            values = np.linspace(start, end, mynumspace)
            if self.space.space[k][0] == 'int':
                mi,ma = self.space.space[k][1][:2]
                if ma - mi <= 1 :
                    self.exhausted.add(k)
                if (ma - mi) <= mynumspace:
                    grp[k] = Range(mi,ma)
                else:
                    grp[k] = np.unique(np.round(values).astype(int))
            else:
                grp[k] = values

        # then make the crossproduct and update the params
        cp = op.maketasks(grp)
        params =[]
        for dic in cp:
            p = self.currentparams.copy()
            p.update(dic)
            params.append(p)


        self.grp = list(grp.keys())
        self.params = params





    def getmax(self):
        dfdf = pd.concat(self.runs)
        return dfdf.sort_values(by='score', ascending=False).iloc[0]['score']

    def print(self):
        dfdf = pd.concat(self.runs)
        # print the params with the highest score
        best_run = dfdf.sort_values(by='score', ascending=False).iloc[0]
        print(best_run)
        #pprint(best_run.drop(['score', 'time', 'datafield']).to_dict())

        # mi = min(dfdf.score[:10])
        # plotscores = [ s if s > mi else mi for s in dfdf.score]
        plt.plot(dfdf.score.cummax().tolist())
        plt.show()
        # plot_params_with_hist(self.params,dfdf)

        return dfdf.sort_values(by='score', ascending=False).head(5)




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




