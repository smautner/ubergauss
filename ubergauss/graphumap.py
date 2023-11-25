import numpy as np
import umap

from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)

def count_non_zeros_per_row(csr_matrix):
    non_zero_counts = []
    for i in range(csr_matrix.shape[0]):
        start_index = csr_matrix.indptr[i]
        end_index = csr_matrix.indptr[i + 1]
        non_zero_counts.append(end_index - start_index)
    return non_zero_counts

def graphumap(csr_matrix, n_dim = 2):
    '''
    csr_matrix is the distance matrix of the graph
    '''
    # initialize matrices
    maxneigh = max(count_non_zeros_per_row(csr_matrix))
    newshape = (csr_matrix.shape[0], maxneigh)
    index = np.full(newshape,-1)
    dist = np.full(newshape,np.inf)

    for i, row in enumerate(csr_matrix):
        order = np.argsort(row.data)
        index[i,:len(order)] = row.indices[order]
        dist[i,:len(order)] = row.data[order]

    myknn = (index, dist, None)
    return umap.UMAP(n_neighbors = index.shape[1], n_components=n_dim, metric='precomputed', precomputed_knn= myknn).fit_transform(csr_matrix)
