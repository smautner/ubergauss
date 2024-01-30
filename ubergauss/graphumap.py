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

def graphumap(adjacency_matrix, n_dim = 2):
    '''
    csr_matrix is the distance matrix of the graph
    '''
    # initialize matrices
    maxneigh = max(count_non_zeros_per_row(adjacency_matrix))
    newshape = (adjacency_matrix.shape[0], maxneigh)
    index = np.full(newshape,-1)
    dist = np.full(newshape,np.inf)

    for i, row in enumerate(adjacency_matrix):
        order = np.argsort(row.data)
        index[i,:len(order)] = row.indices[order]
        dist[i,:len(order)] = row.data[order]

    myknn = (index, dist, None)

    # import structout as so
    # so.heatmap(index)
    # so.heatmap(dist)

    return umap.UMAP(n_neighbors = index.shape[1], n_components=n_dim, metric='precomputed', precomputed_knn= myknn).fit_transform(adjacency_matrix)


import networkx  as nx
def embed_via_nx(adjacency, algo, dim):

    graph = nx.from_scipy_sparse_array(adjacency)

    if algo == 'spring':
        embedding = nx.spring_layout(graph, dim = dim)
    elif algo == 'kamada_kawai':
        embedding = nx.kamada_kawai_layout(graph, dim=dim)
    embedding = np.vstack([ embedding[i] for i in sorted(embedding.keys())])
    return embedding

