import numpy as np

# the spaceship is hyperopt/init
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


def vectorize_parameters(param_dicts, categorical_keys, space=None):
    numeric_features = []
    categorical_features = []
    for d in param_dicts:
        numeric_features.append([v for k, v in d.items() if k not in categorical_keys])
        categorical_features.append([d[k] for k in categorical_keys])
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    categorical_encoded = encoder.fit_transform(categorical_features)
    X = np.hstack([numeric_features, categorical_encoded])

    # Use space object for min-max scaling if provided
    if space is not None:
        # Get numeric parameter keys (non-categorical)
        numeric_keys = [k for k in param_dicts[0].keys() if k not in categorical_keys]

        # Build min/max arrays from space
        min_vals = []
        max_vals = []

        # Numeric parameters first
        for key in numeric_keys:
            min_vals.append(space.space[key][1][0])
            max_vals.append(space.space[key][1][1])

        # Add min/max for one-hot encoded categorical features
        for key in categorical_keys:
            cat_values = space.space[key][1]
            num_categories = len(cat_values)
            # For one-hot encoding, min is 0 and max is 1 for each category
            min_vals.extend([0] * num_categories)
            max_vals.extend([1] * num_categories)

        min_vals = np.array(min_vals)
        max_vals = np.array(max_vals)
    else:
        # Fall back to data-driven scaling
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
    pass

