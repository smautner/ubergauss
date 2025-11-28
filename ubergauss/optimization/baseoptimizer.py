from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from ubergauss import hyperopt as ho
from ubergauss import optimization as op
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pprint import pprint
import structout as so
from hyperopt.pyll.stochastic import sample as hypersample
'''
for ints -> build a model cumsum(occcurance good / occurance bad) ,  then sample accoridngly
for floats -> gaussian sample from the top 50% of the scores
'''
import random



class base():

    def __init__(self, space, f, data, numsample = 16, hyperband = [], mp = True ):
        self.mp = mp
        self.f = f
        self.data = data
        self.numsample = numsample
        self.numsample_proc = numsample
        self.hyperband = hyperband
        if hyperband:
            self.numsample = numsample * 2**(len(hyperband))

        space  = ho.spaceship(space)
        self.params =  [space.sample() for x in range(self.numsample) ]
        self.space = space
        self.scores = []
        self.runs = []
        self.paramgroups = self.getgroups()
        self.key_log ={}

    def getgroups(self):
        keys = {k:[k] for k in self.params[0].keys()}
        return [[k] for k in keys]
        for slave, master in self.space.dependencies.items():
            keys.pop(slave)
            keys[master].append(slave)
        r =  list(keys.values())
        return r

    def hb_pairs(self,x):
        hb =[0]+self.hyperband +[len(self.data)]
        return [(hb[i],hb[i+1])  for i in range(len(hb)-1)]


    def opti(self):
        for i,e in enumerate(self.params):
            e['config_id'] = i

        if self.hyperband:
            df = pd.DataFrame()
            for (start,end) in self.hb_pairs(self.hyperband):
                df2 = op.gridsearch(self.f,
                                    data_list = self.data[start:end],
                                    tasks =self.params)
                df = pd.concat((df,df2))
                lastparams = self.params.copy()
                self.params, df = clean_params(self.params, df) #
            self.df=df
            self.params=lastparams

            # df2 = op.gridsearch(self.f, data_list = self.data[end:],tasks = self.params)
            # self.df = pd.concat((df,df2))

        if not self.hyperband:
            self.df = op.gridsearch(self.f,
                                    data_list = self.data,
                                    tasks =self.params,
                                    mp= self.mp)

            self.df = fix(self.df)


        self.df = self.df.fillna(0)
        self.df = self.df.sort_values(by='score', ascending=False)
        # save the run and get the next parameters
        self.runs.append(self.df)
        self.scores+=self.df.score.tolist()
        self.nuParams()
        return self
        # self.print()

    def getmax(self):
        dfdf = pd.concat(self.runs)
        return dfdf.sort_values(by='score', ascending=False).iloc[0]['score']

    def print(self):
        scr = pd.concat(self.runs)
        so.lprint(scr.score)
        best_run = scr.sort_values(by='score', ascending=False).iloc[0]
        print('Best params:\n', best_run)

    def print_more(self):
        scr = pd.concat(self.runs)
        plt.plot(scr.score.tolist())
        plt.show()
        plot_params_with_hist(self.params, self.df.iloc[:self.numsample_proc], self.key_log)
        plot_term(self.params, self.df.iloc[:self.numsample_proc], self.key_log)



def fix(df):
    '''
    when multiple data items are passed we need to unify them ..
    '''
    if 'data_id' not in df.columns:
        return df

    def f(group):
        # 0:1  so we dont have a series
        result = group.iloc[0:1].copy()
        #result.drop(columns='data_id'.split(), inplace=True)
        result.drop(columns=['data_id'], inplace=True)
        #result['time'] = group['time'].sum()
        result['score'] = group['score'].sum()
        # how to not have the sum of scores but the geomean?
        result['score'] = group['score'].prod()**(1/len(group)) # for geomean
        return result

    return df.groupby('config_id', group_keys = True).apply(f).reset_index(drop=True)


def clean_params(params: list[dict], df: pd.DataFrame):
    # df: somebody concatenated the last run and this run
    # as a result some param_ids) show up multiple times for different datasets with different other_cols

    param_cols = list(params[0].keys())
    other_cols = [col for col in df.columns if col not in param_cols] #task_id data_id score, time

    # group by param_ids
    # sum up the other cols, ignore config_id.
    def f(grp):
        result = grp.iloc[0:1].copy()
        for o in other_cols:
            if o!= 'config_id':result[o] = grp[o].sum()
        return result
    grouped = df.groupby('config_id', group_keys = True).apply(f).reset_index()

    # we want to reduce the param pool by 50%
    new_params = grouped.nlargest(len(params)//2, 'score')[param_cols].to_dict('records')
    return new_params, grouped


def plot_params_with_hist(params, df, log):
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
        plt.title(f"{col} {log.get(col,None)} ")
        fig.tight_layout()
        plt.show()


def plot_term(params, df, log):
    print(f"{ len(df)=}")
    #print(f"{ len(params)=}")
    params = pd.DataFrame(params)


    for col in params.columns:
        if col == "score":
            continue  # Skip the score column itself

        print()
        print(f"{col} {log.get(col,'')}")
        so.scatter(df[col],df.score, columns = 16)
        xlim = min(df[col]), max(df[col])
        so.hist(params[col], bins = 32, xlim = xlim)


