import numpy as np


from sklearn.preprocessing import OneHotEncoder
def df_to_vec(df,space):
    '''
    clean and vectorize
    '''
    df_clean = df.copy()
    cols_to_drop = ['time', 'config_id', 'score']
    df_clean = df_clean.drop(columns=cols_to_drop)
    # 2. vectorize
    param_dicts = df_clean.to_dict(orient='records')
    categorical_keys = [k for k, v in space.space.items() if v[0] == 'cat']
    return vectorize_parameters(param_dicts, categorical_keys)


def vectorize_parameters(param_dicts, categorical_keys):
    numeric_features = []
    categorical_features = []
    for d in param_dicts:
        numeric_features.append([v for k, v in d.items() if k not in categorical_keys])
        categorical_features.append([d[k] for k in categorical_keys])
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    categorical_encoded = encoder.fit_transform(categorical_features)
    X = np.hstack([numeric_features, categorical_encoded])

    # Column-wise min-max scaling of the entire matrix
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # Prevent divide-by-zero
    X = (X - min_vals) / ranges
    return X





# do i want NSGS?





def make_density(df, space):
    '''

    anyway we should look into NSGA first..
    !!!!!!!!!!!!1 this already exists in the gatype file for the clustering..
    # but maybe we can make it better, anyway putting it here will clean things up
    the space stuff worked like this:... so it should be easy to vectorize
    space.space[name] = ['cat',eval(''.join(value_range))]
    '''
    pass



def paretosrt(scores: np.array, densities: np.array):
    # returns non dominated indices



# ok lets start with a visualisation

# first  see above for the stupid clustering -> vectorize...
# now we have a 2d map of the stuff we researched, we can mark the scores even ...
# can mark the selected parents..
# .. how do we integrate this.. lets seeeee
