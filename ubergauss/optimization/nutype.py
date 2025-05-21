from ubergauss import hyperopt as ho
from ubergauss import optimization as op
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pprint import pprint

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
        self.scores = []


    def opti(self):
        self.df = op.gridsearch(self.f, data_list = [self.data],tasks = self.params)
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
        log = {}
        for a in col_names:
            d[a],a_log  = sample(scores, self.df[a].tolist(),self.numsample)
            log[a] = a_log
        d = pd.DataFrame(d)
        self.log = log
        self.params =  d.to_dict(orient='records')


    def print(self):
        print('Best params:', self.df.iloc[-1].to_dict())
        plt.plot(self.scores)
        plt.show()
        plot_params_with_hist(self.params, self.df)
        pprint(self.log)

        # print best params





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
        plt.title(f"{col} vs Score with Distribution Overlay")
        fig.tight_layout()
        plt.show()

def sample(scores, values,n):
    if type(values[0]) == float:
        return floatsample(scores,values,n)
    return intsample(scores,values,n)

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

    return [sample() for _ in range(numsample)], dict(zip(allints, scores))


def floatsample(scores, values, numsample):
    # sort values by scores, keep top 50%
    # scale scores to be between 0 and 100  -> n
    # model = n*associated value for all n
    # calculate mean and std -> sample 100 times

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
    samples = np.random.normal(loc=np.mean(flattened), scale=np.std(flattened), size=numsample)
    # print mean and std
    log = f"mean: {np.mean(flattened)}, std: {np.std(flattened)}"
    return samples, log


