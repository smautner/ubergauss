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

class nutype:

    def __init__(self, space, f, data, numsample = 16):
        self.f = f
        self.data = data
        self.numsample = numsample

        space  = ho.spaceship(space)
        self.params =  [space.sample() for x in range(self.numsample) ]
        self.samplers = make_samplers(self.params, space)
        self.scores = []


    def opti(self):
        self.df = op.gridsearch(self.f, data_list = self.data,tasks = self.params)
        self.df = fix(self.df)
        # drop nans
        self.df = self.df.dropna()
        self.df = self.df.sort_values(by='score', ascending=True)
        self.scores+=self.df.score.tolist()
        self.nuParams()
        # self.print()


    def nuParams(self):
        scores = self.df.score.tolist()
        # get all the column names except time, score and datafield
        col_names = [col for col in self.df.columns if col not in ['time', 'score', 'datafield']]
        d = {}

        for a in col_names:
            d[a]  = self.samplers[a].sample(scores, self.df[a].tolist(),self.numsample)
        d = pd.DataFrame(d)
        self.params =  d.to_dict(orient='records')


    def print(self):
        print('Best params:', self.df.iloc[-1].to_dict())
        plt.plot(self.scores)
        plt.show()
        plot_params_with_hist(self.params, self.df,self.samplers)

        # print best params




def make_samplers(params:dict, spaceship) -> dict:
    samplers = {}
    for k,v in params[0].items():
        orig = partial(hypersample, spaceship.space[k])
        if type(v) == float:
            sampler = floatsampler(orig)
        else:
            sampler = intsampler(orig)
        samplers[k] = sampler
    return samplers


class intsampler():
    def __init__(self, s):
        self.sampler = s
        self.log = 'i am just an intsampler'
    def sample(self,scores, values, numsample):
        r, self.log =  intsample(scores,values,numsample)
        return r

def intsample(scores, values, numsample):
    # take top 40% and bottom 40%
    scores = np.array(scores)
    values = np.array(values)
    sorted_indices = np.argsort(scores)[::-1]
    top_40 = sorted_indices[:int(len(scores) * 0.4)]
    bottom_40 = sorted_indices[int(len(scores) * 0.6):]

    # calc probability for each integer:
    # allints = unique(top40)
    # freqscore = [ score(int) for allints]
    # score is the occurance in top40 / occ in bottom +1
    allints = np.unique(values[top_40])
    def getscore(i):
        top_count = np.sum(values[top_40] == i)
        bottom_count = np.sum(values[bottom_40] == i) + 1
        return top_count / bottom_count
    scores = np.array([getscore(i) for i in allints])

    # now we can make a cumsum of the scores, scale up a random.random and choose one of the scores

    cum_scores = np.cumsum(scores)
    total_score = cum_scores[-1]
    def sample():
        r = np.random.uniform(0, total_score)
        chosen_index = np.searchsorted(cum_scores, r)
        return allints[chosen_index]

    return [sample() for _ in range(numsample)], Zip(allints, scores)

class floatsampler():
    def __init__(self, s):
        self.sampler = s
        self.log = 'asd'

    def sample(self,scores, values, numsample):
        self.log = need_sampler(scores,values)
        if self.log < 1: # basically turn it of as it doesnt help... maybe use correlation after all??
            self.sampler = learn_float_sampler(scores,values)
        return [self.sampler() for _ in range(numsample)]

def need_sampler(scores,values):
    # sort score and values by scores
    # throw away the middle 20%
    # then label topscores 1, bottomscores 0  -> y
    #return sum(y == np.roll(y,-1))-2/len(y)
    sv = Zip(scores,values)
    scoresort = sorted(sv, key = lambda x:x[0])
    p40 = int(len(sv)*.4)
    score1 = [ (0,v) for s,v in scoresort[:p40] ]
    score1+= [ (1,v) for s,v in scoresort[-p40:]]
    valsort = sorted(score1, key = lambda x:x[1])
    y = np.array([ii for ii,_ in valsort])
    score = (np.sum(y == np.roll(y, -1)) -2) / (len(y)-2) # 2 misses are allowed :)
    # 0 -> all the same  -> i shoudl resample
    # 1 -> all different -> dont resample
    return score

def learn_float_sampler(scores,values):
        scores = np.array(scores)
        values = np.array(values)

        sorted_indices = np.argsort(scores)[::-1]
        topat = int(len(scores) * 0.4)
        top_half = sorted_indices[:topat]
        top_scores = scores[top_half]
        top_values = values[top_half]

        min_score = top_scores.min()
        max_score = top_scores.max()
        if max_score == min_score:
            scaled_scores = np.full_like(top_scores, 100.0)
        else:
            scaled_scores = 100 * (top_scores - min_score) / (max_score - min_score)
        flattened = [v for s, v in zip(scaled_scores, top_values) for _ in range(int(s))]

        # flattened = top_values
        m,s = np.mean(flattened),np.std(flattened)
        # print(f"{m=} {s=} {values=}")
        samples = lambda: np.random.normal(m,s)
        # print mean and std
        return samples



def fix(df):
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

def plot_params_with_hist(params, df, samp):
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
        plt.title(f"{col} -- {samp[col].log}")
        fig.tight_layout()
        plt.show()


