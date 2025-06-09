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




class base():

    def __init__(self, space, f, data, numsample = 16, hyperband = [] ):
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
        if self.hyperband:
            df = pd.DataFrame()
            for (start,end) in self.hb_pairs(self.hyperband):
                df2 = op.gridsearch(self.f, data_list = self.data[start:end],tasks = self.params)
                df = pd.concat((df,df2))
                lastparams = self.params.copy()
                self.params, df = clean_params(self.params, df)
            self.df=df
            self.params=lastparams

            # df2 = op.gridsearch(self.f, data_list = self.data[end:],tasks = self.params)
            # self.df = pd.concat((df,df2))

        if not self.hyperband:
            self.df = op.gridsearch(self.f, data_list = self.data,tasks = self.params)
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
        plt.plot(scr.score.tolist())
        plt.show()

        plot_params_with_hist(self.params, self.df.iloc[:self.numsample_proc], self.key_log)
        plot_term(self.params, self.df.iloc[:self.numsample_proc], self.key_log)

        # print best params



def fix(df):
    # Identify columns that define a unique parameter combination (all except 'score' and 'datafield')
    param_cols = [col for col in df.columns if col not in ['score', 'datafield' ,'time']]

    # Calculate the average score for each parameter combination across all datafields
    avg_scores = df.groupby(param_cols)['score'].sum().reset_index()

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
        so.scatter(df[col],df.score)
        so.hist(params[col], bins = 20)


        # fig.scatter(df[col], df.score, label='df data')

        ## plotille..
        #import plotille
        #fig = plotille.Figure()
        #fig.width = 60
        #fig.height = 3
        ##fig.set_x_limits(min_=-3, max_=3)
        ##fig.set_y_limits(min_=-1, max_=1)
        ##fig.color_mode = 'byte'
        #fig.scatter(df[col], df.score, label='df data')
        ## sns.histplot(params[col], ax=ax2, color='gray', alpha=0.3, bins=20, label='Distribution')
        #v = params[col]
        #y, X  = np.histogram(v,50)
        #X = [np.mean(X[x:x+1]) for x in range(len(X)-1)]
        #fig.scatter(X,y, label='params')
        #print(fig.show(legend=True))



def clean_params(params:dict, df:pd.DataFrame) -> tuple:
    '''
    this will replace clean params:
    - df contains multple entried of the params, so we can groupby params.keys  and sum up the columns not in the dict (score datafield time)
    - then we return the the top scoring len(params)//2 rows as the new datadict (no score datafield and time)
    '''

    # Param columns = keys of dict
    param_cols = list(params[0].keys())

    # All other columns will be aggregated (sum)
    agg_cols = [col for col in df.columns if col not in param_cols]

    # 1. Group and sum by param columns
    grouped = df.groupby(param_cols)[agg_cols].sum().reset_index()

    # 2. Sort by score to pick top N
    num_to_keep = max(1, len(params) // 2)
    top_groups = grouped.sort_values(by='score', ascending=False).head(num_to_keep)

    # 3. Extract top param dicts
    top_param_dicts = top_groups[param_cols].to_dict(orient='records')

    return top_param_dicts, grouped



def clean_params_old(df: pd.DataFrame) -> tuple:
    # Compute average score per group
    print(f"{ len(df)=}")
    group_cols = [col for col in df.columns if col not in ['score', 'datafield' ,'time']]
    group_scores = df.groupby(group_cols)['score'].mean().reset_index()

    # Sort by score descending and select top 50%
    top_groups = group_scores.sort_values(by='score', ascending=False)
    top_groups = top_groups.head(len(top_groups)//2)

    # Merge to filter original DataFrame
    #mask = df[group_cols].apply(tuple, axis=1).isin( top_groups[group_cols].apply(tuple, axis=1))
    filtered_df = df.merge(top_groups[group_cols], on=group_cols, how='inner')

    list_of_dicts = filtered_df[group_cols].drop_duplicates().to_dict(orient='records')
    print(f"{ len(filtered_df)=}")
    print(f"{ len(list_of_dicts)=}")
    # breakpoint()
    #return  list_of_dicts, filtered_df, df[~mask]
    return  list_of_dicts, filtered_df#, df[~mask]




