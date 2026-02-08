import numpy as np
import pandas as pd

# the spaceship is hyperopt/init
from sklearn.preprocessing import OneHotEncoder


def df_to_vec(df, space):
    """
    clean and vectorize
    """
    if df is None or len(df) == 0:
        return np.array([])
    # REMOVE TRASH
    df_clean = df.copy()
    # assert  isinstance(df_clean, pd.DataFrame), breakpoint()
    cols_to_drop = ["time", "config_id", "score", "data_id"]
    cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)


    if not df_clean.shape[0]:
        return np.array([])

    cat_keys = [k for k, v in space.space.items() if v[0] == "cat"]
    num_keys = [k for k in df_clean.columns if k not in cat_keys]

    # Encoding
    if cat_keys:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        # Ensure all categories from space are known to the encoder
        assert False, "implement this later..."
        categories = [space.space[k][1] for k in cat_keys]
        encoder.categories = categories
        encoded_cats = encoder.fit_transform(df_clean[cat_keys])
        X = np.hstack([df_clean[num_keys].values, encoded_cats])
    else:
        X = df_clean[num_keys].values

    # Scaling
    for i, name in enumerate(num_keys):
        # column should be i, in the step before we sorted accordingly
        minv, maxv = space.space[name][1]
        X[:, i] = (X[:, i] - minv) / (maxv - minv)

    return X




# do i want NSGS?

def make_density(df, space):
    """

    anyway we should look into NSGA first..
    !!!!!!!!!!!!1 this already exists in the gatype file for the clustering..
    # but maybe we can make it better, anyway putting it here will clean things up
    the space stuff worked like this:... so it should be easy to vectorize
    space.space[name] = ['cat',eval(''.join(value_range))]
    """
    pass


def paretosrt(scores: np.array, densities: np.array):
    # returns non dominated indices
    pass


import umap
import matplotlib.pyplot as plt



def prepare_vectors(runs, params,parents, space):
    """-> vector dict for  runs (car, cdr) params parents """
    vector_dict = {}
    cdr  = pd.concat(runs[:-1]) if len(runs) > 1 else pd.DataFrame()
    car  = runs[-1].copy()
    params = pd.DataFrame(params)
    parents = pd.DataFrame(parents)
    return {name: df_to_vec(df, space) for df, name in zip([car, cdr, params, parents], "car cdr params parents".split())}



def fit_or_transform_umap(vectors, reducer=None):
    """Fit UMAP or transform with existing reducer."""



    # if reducer is None:
    #     class FakeReducer:
    #         def transform(self, x):
    #             if len(x) == 0:
    #                 return np.array([]).reshape(0, 2)
    #             return x[:, :2]
    #     reducer = FakeReducer()

    if reducer is None:
        breakpoint()
        reducer = umap.UMAP(n_neighbors=10, random_state=42)
        reducer.fit(vectors['car'])

    return reducer, {k: reducer.transform(v) for k,v in vectors.items()}




def render_umap_plot(layers, score):
    """Render the complete UMAP plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # """-> vector dict for  runs (car, cdr) params parents """


    # Old runs - small black dots
    if 'cdr' in layers:
        ax.scatter(*layers['cdr'].T, c='black', s=10, alpha=0.3, label='Old runs')

    # Last run - viridis
    if 'car' in layers:
        scatter = ax.scatter(
            *layers['car'].T,
            c=score,
            cmap='viridis',
            alpha=0.7,
            s=60,
            label='Last run'
        )
        cbar = plt.colorbar(scatter, ax=ax)
        # cbar.set_label('score')

    # Selected parents - red hollow circles
    if 'parents' in layers:
        ax.scatter(
            *layers['parents'].T,
            facecolors='none',
            edgecolors='red',
            s=150,
            linewidths=2,
            marker='o',
            label='Selected parents',
            alpha=1.0
        )

    # New generation - red dots
    if 'params' in layers:
        ax.scatter(
            *layers['params'].T,
            c='red',
            s=30,
            marker='o',
            alpha=0.8,
            label='Next generation'
        )

    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_title('Optimization Landscape')
    ax.legend()
    plt.show()


def plot_umap_visualization(runs, params, space, reducer=None, parents= None):
    """
    Visualize optimization landscape using UMAP.

    Parameters:
    - runs: list of dataframes with historical runs
    - params: list of parameter dicts for current generation
    - space: search space object
    - reducer: UMAP reducer (learned on first call, reused thereafter)
    - parent_indices: selected parents

    Returns:
    - reducer: UMAP reducer for reuse
    """
    vectors = prepare_vectors(runs, params, parents, space)
    """-> vector dict for  runs (car, cdr) params parents """
    reducer, layers = fit_or_transform_umap(vectors, reducer)
    scores = runs[-1]["score"].values
    render_umap_plot(layers, scores)
    return reducer

