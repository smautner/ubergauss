import numpy as np
import sklearn


def transform(distance_matrix, k=10, algo = 2, startfrom = 1):
    """
    0 -> do nothing
    1 -> normalize by norm
    2 -> csls
    3 -> ls
    4 -> nicdm
    """

    # assert distance_matrix[0][0] < distance_matrix[0][1], 'checking if this is a distance matrix'

    if algo == 0:
        return distance_matrix
    if algo == 1:
        return sklearn.preprocessing.normalize(distance_matrix, axis = 0)

    # if algo == 2:
    #     return MP(distance_matrix, k + 15)

    funcs = [csls_, ls, nicdm, ka]
    f = funcs[algo-2]

    n = distance_matrix.shape[0]
    # scaled_distances = distance_matrix.copy()
    startfrom = 1
    knn = np.partition(distance_matrix, k+startfrom , axis=1)[:, :k+startfrom ]  # +1 to account for self
    knn = np.sort(knn, axis = 1)
    knn = knn[:,startfrom :].mean(axis = 1)

    # Apply scaling
    for i in range(n):
        for j in range(n):
            v = distance_matrix[i,j]
            distance_matrix[i,j]  =  f(v,knn[i],knn[j])
    return distance_matrix


def csls_(v,i,j):
    return v*2 -i -j
def ls(v,i,j):
    return 1- np.exp(- v**2/(i*j) )
def nicdm(v,i,j):
    return v /  np.sqrt(i*j)
def ka(v,i,j):
    return v / i +  v/j

# def another(v,i,j):
#     return v * j ** .5
