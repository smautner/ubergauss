from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.example_libraries.optimizers import adam
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.decomposition import TruncatedSVD, PCA

from jax import config
config.update("jax_debug_nans", False)
config.update("jax_enable_x64", True)

def toCooTrip(X):
    coo=coo_matrix(X)
    i,j =coo.row, coo.col
    for ii,jj in zip(i,j):
        # if the other one has a signal or we are on the diagonal, remove current
        # if its symmetrical we only need 1 in the opti targets, also the diagonal is boring for opti
        if X[jj,ii] != 0 or jj==ii:
            X[ii,jj] = 0
    coo=coo_matrix(X)
    coo.eliminate_zeros()
    return Map(jnp.array, [coo.row, coo.col, coo.data])


def loss_justdistance(embedding, triplets):
    distance_pred = jnp.linalg.norm(embedding[triplets[0]] - embedding[triplets[1]], axis=1)
    losses = (distance_pred - triplets[2]) ** 2
    return jnp.mean(losses)

def loss(embedding, good_trip=None, bad_trip=None, w=(1,1)):
    # distance_pred = jnp.linalg.norm(embedding[good_trip[0]] - embedding[good_trip[1]], axis=1)
    nearpoint_dist = (embedding[good_trip[0]] - embedding[good_trip[1]]) ** 2; nearpoint_dist = nearpoint_dist.sum(axis=1)
    loss_near = (nearpoint_dist+1)/(nearpoint_dist+10)

    farpoint_dist = (embedding[bad_trip[0]] - embedding[bad_trip[1]]) ** 2; farpoint_dist = farpoint_dist.sum(axis=1)
    # dp2 = jnp.linalg.norm(embedding[bad_trip[0]] - embedding[bad_trip[1]], axis=1)
    loss_far = 1/(farpoint_dist+1)

    loss_near = jnp.sum(loss_near)
    loss_far = jnp.sum(loss_far)
    # jax.debug.print("{}", ln+lf)
    return loss_near*w[0]+loss_far*w[1]


def optimize(embedding, step_size = .2, steps = 100, **lossargs):
    # Optimizer setup
    opt_init, opt_update, get_params = adam(step_size)
    opt_state = opt_init(embedding)
    grad_loss = grad(loss)
    # Training loop
    for i in range(steps):
        gradients = grad_loss(get_params(opt_state), **lossargs)
        opt_state = opt_update(i, gradients, opt_state)

    return get_params(opt_state)

def embed(X,n_components = 2):
    X,Xbad=X
    good_trip = toCooTrip(X)
    bad_trip = toCooTrip(Xbad)
    # tsvd for init, should be nice as it plays well with sparse data
    # embedding = jnp.array(TruncatedSVD(n_components=n_components).fit_transform(X)) # L shaped result
    embedding = np.random.rand(X.shape[0],2)
    # embedding = PCA(n_components=n_components).fit_transform(X.toarray())


    #embedding = optimize(embedding,good_trip=good_trip, bad_trip = bad_trip, w = (2,1))
    embedding = optimize(embedding,good_trip=good_trip, bad_trip = bad_trip, w = (1,5))
    embedding = optimize(embedding,good_trip=good_trip, bad_trip = bad_trip, w = (2,1))

    return embedding




if __name__ == f"__main__":
    X = csr_matrix([[0,4.8,2],
                    [4.8,0,3.8],
                    [2,3.8,0]])

    print(embed(X,n_components = 2))
