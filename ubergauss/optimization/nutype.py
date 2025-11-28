from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from functools import partial
import pandas as pd
import numpy as np
from hyperopt.pyll.stochastic import sample as hypersample
from ubergauss.optimization import baseoptimizer
from sklearn.neighbors import NearestNeighbors
from scipy.stats import binom
from sklearn import feature_selection
import random
'''
basically TPE style optimization per parameter vs score.
distributions are updated after a test is passed:
    - i remove categoricals base on p-values
    - mutual info threshold for continuous variables
'''


def create_and_update_param_samplers(paramgroups, space, runs):
    individual_samplers = []
    for p_group in paramgroups:
        param_name = p_group[0]
        sampler = mks(space, param_name)
        individual_samplers.append(sampler)

    for sampler in individual_samplers:
        sampler.update(runs)
    return {s.name: s for s in individual_samplers}



class nutype(baseoptimizer.base):

    def nuParams(self):
        '''
        first the non dependants can be done as usual,
        then for the dependants, they need subsamplers , so a sampler returns a dict v:k... in the end i can combine all the dicts...
        '''

        if not hasattr(self, 'samplers'):
            self.samplers = [mks(self.space,p[0]) for p in self.paramgroups]

        # if not hasattr(self, 'carry'):
        #     self.carry = pd.DataFrame()
        # data = pd.concat((self.carry,self.df))
        # data = pd.concat((self.carry,self.df))
        # data = data.sort_values(by='score', ascending=False)

        # print(self.df[:8])
        for s in self.samplers:
            self.key_log[s.name] = s.update(self.runs)
            print(s.name, self.key_log[s.name])
        self.params = [self.sample() for _ in range(self.numsample)]

        # c = int(self.numsample*.4)
        # self.carry = data.head(c).copy()


    def sample(self):
        d={}
        for s in self.samplers:
            d.update(s.sample())
        return d


def mks(space,key):
    if space.space[key][0]=='cat':
        return CS(space,key)
    if space.space[key][0]=='float':
        return FS(space,key)
    if space.space[key][0]=='int':
        return IS(space,key)
class Simple():
    def __init__(self, space,key):
        self.name=key
        self.par = space.space[key]
        self.sample_f = partial(hypersample, space.hoSpace[key])
    def sample(self):
        return {self.name:self.sample_f()}





# choices = ['apple', 'banana', 'cherry']
# weights = [0.1, 0.2, 0.7] # Probabilities for each choice
# chosen_item = random.choices(choices, weights, k=1)

class CS(Simple):
    def update(self,runs):
        # df = runs[-1]
        # comment = p_values(df[self.name], df.score)
        # comment = p_values_allruns(runs, self.name)
        comment = p_values_allruns(runs, self.name)
        # self.sample_f = partial(random.choice, [k for k,v in comment.items() if v < .9])
        self.sample_f = partial(choice, comment)
        return comment

def choice(p_dict):
    return random.choices(list(p_dict.keys()), weights=[ 1- v for v in p_dict.values()], k=1)[0]



def p_values_allruns_alternative(runs, cat):
    """
    Alternative to p_values_allruns: ranks instances implicitly, then for each
    instance compares its score to a random instance from a different category,
    counting wins across all datasets.
    """
    category_wins = {}
    category_trials = {}

    for df in runs:
        if df.empty:
            continue

        unique_categories = df[cat].unique()

        for current_cat_value in unique_categories:
            current_category_df = df[df[cat] == current_cat_value]
            other_categories_df = df[df[cat] != current_cat_value]

            if current_category_df.empty or other_categories_df.empty:
                # Cannot make a meaningful comparison if either group is empty
                continue

            wins_for_this_cat_in_run = 0
            trials_for_this_cat_in_run = 0

            # Compare each instance in the current category to a randomly chosen instance
            # from a different category within the same run.
            for _, current_instance in current_category_df.iterrows():
                # Randomly select one instance from other categories
                random_other_instance = other_categories_df.sample(n=1).iloc[0]

                if current_instance['score'] > random_other_instance['score']:
                    wins_for_this_cat_in_run += 1
                trials_for_this_cat_in_run += 1

            # Aggregate wins and trials for this category across all runs
            category_wins[current_cat_value] = category_wins.get(current_cat_value, 0) + wins_for_this_cat_in_run
            category_trials[current_cat_value] = category_trials.get(current_cat_value, 0) + trials_for_this_cat_in_run

    results = {}
    overall_p_null = 0.5  # Null hypothesis: 50% chance of winning in a random comparison

    for category, k in category_wins.items():
        n = category_trials[category]
        if n == 0:
            # If no comparisons were made for this category, assign a neutral p-value
            results[category] = 1.0
            continue

        # Calculate p-value using the binomial survival function (P(X >= k))
        # binom.sf(k-1, n, p) gives P(X >= k)
        p_value = binom.sf(k - 1, n, overall_p_null)
        results[category] = p_value

    return results


def p_values_allruns(runs, cat):
    # concatenate all the categories
    x = pd.concat([r[cat] for r in runs]).values
    y = pd.concat([r.score> np.median(r.score) for r in runs]).values
    return calculate_p_values(x, y)



def p_values(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    success = y >= np.median(y)
    return calculate_p_values(x, success)

def calculate_p_values(x, success):
    overall_p = .5
    # Compute p-value for each category
    results = {}
    for category in np.unique(x):
        category_success = success[x == category]
        n = len(category_success)
        k = np.sum(category_success)
        p_value = binom.sf(k, n, overall_p)
        results[category] = p_value
    return results




class FS(Simple):
    def update(self, runs):
        # Concatenate all runs into a single DataFrame
        all_data = pd.concat(runs, ignore_index=True)

        # Sort by score in descending order
        all_data = all_data.sort_values(by='score', ascending=False)

        # Get the top 15 of the data
        top_data = all_data.head(15)

        # Get the values for the current parameter from the top data
        top_values = top_data[self.name].values

        # Ensure we have at least two values to calculate min/max
        if len(top_values) < 2:
            # Fallback to original space if not enough data
            self.sample_f = partial(hypersample, self.space.hoSpace[self.name])
            return "Not enough data for informed sampling"

        # Define the sampling range
        min_val = np.min(top_values)
        max_val = np.max(top_values)

        # Set the sample function to a uniform distribution within this range
        self.sample_f = partial(np.random.uniform, min_val, max_val)

        return f"Sampling uniformly between {min_val:.2f} and {max_val:.2f}"

class FS_old(Simple):
    def update(self,runs):
        df = runs[-1]
        mutual_info = should_learn_float(df[self.name], df.score)
        if mutual_info > .2:
            top = .5
            if mutual_info > .3: top -= .1
            if mutual_info > .4: top -= .1
            if mutual_info > .5: top -= .1
            self.sample_f = learn_float_sampler(df.score,df[self.name], learntop = top)
        return mutual_info

def should_learn_float(values, score):
    scores = score.to_numpy()
    values = values.values.reshape(-1, 1)
    log = feature_selection.mutual_info_regression(values, scores, n_neighbors=2, discrete_features=False)
    return log[0]

def learn_float_sampler(scores,values,learntop=.4):
        scores = np.array(scores)
        values = np.array(values)
        weights = np.argsort(scores)
        vals = [values[i] for i in weights[int(len(scores)*learntop):]]

        # return partial(np.random.uniform,np.min(vals)*.9,np.max(vals)*1.1)
        return partial(np.random.normal,np.mean(vals),np.std(vals))
        # return partial(np.random.normal,np.mean(vals),np.std(vals))
        # flattened = [v for s, v in zip(weights, values) for _ in range(int(s))]
        # m,s = np.mean(flattened),np.std(flattened)*.5



class IS(Simple):
    def update(self, runs):
        # Use the logic from FS to determine the sampling range
        all_data = pd.concat(runs, ignore_index=True)
        all_data = all_data.sort_values(by='score', ascending=False)
        top_data = all_data.head(15)
        top_values = top_data[self.name].values

        if len(top_values) < 2:
            # Fallback to original space if not enough data
            self.sample_f = partial(hypersample, self.space.hoSpace[self.name])
            return "Not enough data for informed sampling"

        min_val = np.min(top_values)
        max_val = np.max(top_values)

        # Create a sampler that generates integers within the derived range
        # and respects the original parameter's min/max bounds
        par_min, par_max = self.par[1][:2] # Assumes self.par is like ('int', (min, max))

        def int_sampler():
            while True:
                # Sample from a uniform distribution within the learned float range
                # Then round to the nearest integer
                v = int(np.random.uniform(min_val, max_val) + 0.5)
                # Ensure the sampled value is within the original parameter's bounds
                if par_min <= v <= par_max:
                    return v

        self.sample_f = int_sampler
        return f"Sampling integers uniformly between {int(min_val)} and {int(max_val)} (within {par_min}-{par_max})"



class IS_old(Simple):
    def update(self,runs):
        '''
        1. check if we need to update the sampler.
        2. this will generate a comment that we might want to return
        3. then we update the sampler if necessary
        '''
        df = runs[-1]
        mutInfo = should_learn_float(df[self.name], df.score)
        if mutInfo > .2:
            sample_f = learn_float_sampler(df.score,df[self.name])

            def int_sampler():
                while True:
                    v = int(sample_f()+.5)
                    par = self.par[1]
                    if par[0]<= v <= par[1]:
                        return v
            self.sample_f = int_sampler
        return mutInfo
























def learn_cat_sampler(scores, values):
    '''
    this actually works nicely. looks at the worst and best instances and samples accordingly
    '''
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
        scr =  top_count / bottom_count
        return scr
    scores = np.array([getscore(i) for i in allints])
    # print(dict(zip(allints, scores)))

    # now we can make a cumsum of the scores, scale up a random.random and choose one of the scores
    cum_scores = np.cumsum(scores)
    total_score = cum_scores[-1]
    def sample():
        r = np.random.uniform(0, total_score)
        chosen_index = np.searchsorted(cum_scores, r)
        return allints[chosen_index]
    return sample






def should_learn_float_old(x, y, n_neighbors=1):
    """
    Predicts y using average of y-values of nearest neighbors in x,
    and returns the normalized prediction error.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Fit Nearest Neighbors model (excluding the point itself)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto')
    nn.fit(x)
    distances, indices = nn.kneighbors(x)
    pred_y = np.array([
        np.mean(y[indices[i][1:]]) for i in range(len(y))
    ])

    # Compute Mean Squared Error
    mse = np.mean((y - pred_y) ** 2)

    # Expected squared difference between any two y values (normalizer)
    diffs = y[:, None] - y[None, :]
    expected_squared_diff = np.mean(diffs ** 2)

    # Normalized error
    normalized_error = mse / expected_squared_diff if expected_squared_diff != 0 else np.nan

    return normalized_error





def learn_float_sampler_old(scores,values):
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


def need_sampler(scores,values):
    '''
    this was my ring checker
    '''
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


class Samplerr():
    def __init__(self, space, keys):
        self.name = keys[0]
        self.mainsampler = Simple(space,keys[0])
        self.sub = {cat:Simple(space,keys[1]) for cat in space.space[keys[0]][1] }
        # orig = partial(hypersample, spaceship.hoSpace[k])

    def learn(self, df):
        self.mainsampler.learn(df)
        for e in self.sub:
            df2 = df[df[self.name] == e]
            if len(df2)>2:
                self.sub[e].learn(df2)

    def sample(self):
        r = self.mainsampler.sample()

        r.update(self.sub[list(r.values())[0]].sample())
        return r



def Sampler(space, keys):
    if len(keys) ==1:
        return  Simple(space, keys[0])
    else:
        return  Samplerr(space, keys)




from deap import benchmarks
from scipy.stats import gmean
def test():
    '''
    just run nutype on the benchmark problems... and report average score
    '''
    functions = {
        "sphere": benchmarks.sphere,
        # "rastrigin": benchmarks.rastrigin,
        # "rosenbrock": benchmarks.rosenbrock,
        # "ackley": benchmarks.ackley,
        # "schwefel": benchmarks.schwefel,
        # "griewank": benchmarks.griewank,
        # "h1": benchmarks.h1  # Multi-modal function
    }
    space  = '\n'.join([f'x{i} -5 5' for i in range(5)])
    for name, func in functions.items():
        print(name)
        def f(**params):
            return -func(list(params.values()))[0]

        o = nutype(space, f, data=[[]], numsample=32)
        [o.opti() for i in range(10)]
        o.print()
        print(o.runs[-1])




