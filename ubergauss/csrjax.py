from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.example_libraries.optimizers import adam
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.decomposition import TruncatedSVD

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


def loss(embedding, triplets):
    distance_pred = jnp.linalg.norm(embedding[triplets[0]] - embedding[triplets[1]], axis=1)
    losses = (distance_pred - triplets[2]) ** 2
    return jnp.mean(losses)


def embed(X,n_components = 2):
    coo_triplets = toCooTrip(X)
    # tsvd for init, should be nice as it plays well with sparse data
    embedding = TruncatedSVD(n_components=n_components).fit_transform(X)


    # Optimizer setup
    step_size = 0.01
    opt_init, opt_update, get_params = adam(step_size)
    opt_state = opt_init(embedding)
    grad_loss = grad(loss)

    # Training loop
    for i in range(450):
        gradients = grad_loss(get_params(opt_state), coo_triplets)
        opt_state = opt_update(i, gradients, opt_state)

    return get_params(opt_state)




if __name__ == f"__main__":
    X = csr_matrix([[0,4.8,2],
                    [4.8,0,3.8],
                    [2,3.8,0]])

    print(embed(X,n_components = 2))
