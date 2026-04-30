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

def loss_orig(embedding, good_trip=None, bad_trip=None, w=(1,1)):
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


def loss(embedding, good_trip=None, bad_trip=None, w=(1, 1)):
    # Attraction: Force points together (squared distance)
    d_pos_sq = jnp.sum((embedding[good_trip[0]] - embedding[good_trip[1]])**2, axis=1)
    # Using 1/(1+d^2) style weights or simple mean
    loss_near = jnp.mean(d_pos_sq)
    # Repulsion: Force points apart using an inverse-square or log law
    # This ensures points keep moving even if they are > 1.0 apart
    d_neg_sq = jnp.sum((embedding[bad_trip[0]] - embedding[bad_trip[1]])**2, axis=1) + 1e-6
    loss_far = jnp.mean( 1/ d_neg_sq)
    # loss_far = - jnp.mean( d_neg_sq)
    return w[0] * loss_near + w[1] * loss_far



def optimize(embedding, step_size = .2, steps = 50, **lossargs):
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


def embed_adjacency(adj, n_components=2, steps=100):
    # Split adjacency into close (1) and far (2) relationships
    X_good = adj.copy()
    X_good.data = (X_good.data == 1).astype(float)
    X_bad = adj.copy()
    X_bad.data = (X_bad.data == 2).astype(float)

    good_trip = toCooTrip(X_good)
    bad_trip = toCooTrip(X_bad)

    # embedding = jnp.array(np.random.normal(0, 0.1, (adj.shape[0], n_components)))
    # embedding = np.random.rand(adj.shape[0],n_components)

    # Better initialization
    # embedding = TruncatedSVD(n_components=n_components).fit_transform(adj)  do this byt remove the 2s from the adj

    X_init = adj.copy()
    X_init.data[X_init.data == 2] = 0
    X_init.eliminate_zeros()
    # embedding = TruncatedSVD(n_components=n_components).fit_transform(X_init)
    embedding = PCA(n_components=n_components).fit_transform(X_init)
    embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
    embedding = jnp.array(embedding)


    # Optimization passes
    embedding = optimize(embedding, step_size=0.1, steps=steps, good_trip=good_trip, bad_trip=bad_trip, w=(1, 1))
    # embedding = optimize(embedding, step_size=0.01, steps=steps, good_trip=good_trip, bad_trip=bad_trip, w=(2, 1))
    # embedding = optimize(embedding, step_size=0.2, steps=steps, good_trip=good_trip, bad_trip=bad_trip, w=(1, 2))
    # embedding = optimize(embedding, step_size=0.1, steps=steps, good_trip=good_trip, bad_trip=bad_trip, w=(2, 1))

    return embedding




def test_embedder():
    # 0 and 1 are close, 0 and 2 are far, 1 and 2 are far
    data = [1, 2, 1, 2, 2, 2]
    row = [0, 0, 1, 1, 2, 2]
    col = [1, 2, 0, 2, 0, 1]
    adj = csr_matrix((data, (row, col)), shape=(3, 3))

    res = embed_adjacency(adj)
    print("Embedding result:\n", res)

    d01 = jnp.linalg.norm(res[0] - res[1])
    d02 = jnp.linalg.norm(res[0] - res[2])
    print(f"Distance 0-1 (close): {d01:.4f}, Distance 0-2 (far): {d02:.4f}")
    assert d01 < d02, "Close points should be nearer than far points"


if __name__ == f"__main__":
    test_embedder()
    X = csr_matrix([[0,4.8,2],
                    [4.8,0,3.8],
                    [2,3.8,0]])

    print(embed(X,n_components = 2))
