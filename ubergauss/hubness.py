import numpy as np
import sklearn




def transform(X,Y, k=10, algo = 2,  kstart = 1, metric = 'cosine'):
    X = sklearn.metrics.pairwise_distances(X, Y, metric=metric)
    return justtransform(X, k=k, algo = algo,  kstart = kstart)

def justtransform(distance_matrix, k=10, algo = 2,  kstart = 1):
    '''
    k for the neighbors
    algo for the algorithm
    kstart use k neighbors starting from here, to not have the
    '''
    k= int(k)
    algo = int(algo)

    if algo == 0:
        return distance_matrix
    if algo == 1:
        return sklearn.preprocessing.normalize(distance_matrix, axis = 0)
    funcs = [csls_, ls, nicdm, ka]
    f = funcs[algo-2]

    n = distance_matrix.shape[0]
    knn = np.partition(distance_matrix, k+kstart , axis=1)[:, :k+kstart ]  # +1 to account for self
    knn = np.sort(knn, axis = 1)
    knn = knn[:,kstart :].mean(axis = 1)

    # Apply scaling
    for i in range(n):
        for j in range(n):
            v = distance_matrix[i,j]
            distance_matrix[i,j]  =  f(v,knn[i],knn[j])
    return distance_matrix

def transform_experiments_SLOW(distance_matrix, k=10, algo = 2, kiezbug = 0, kiezpresort = 50):
    """
    0 -> do nothing
    1 -> normalize by norm
    2 -> csls
    3 -> ls
    4 -> nicdm
    """

    # assert distance_matrix[0][0] < distance_matrix[0][1], 'checking if this is a distance matrix'
    k= int(k)
    algo = int(algo)
    kiezbug = int(kiezbug)
    kiezpresort = int(kiezpresort)

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
    knn_good = knn[:,startfrom :].mean(axis = 1)
    if kiezbug:
        knn_bad =  knn[:,:-1].mean(axis = 1)
    else:
        knn_bad = knn_good

    # MAKING THINGS WORSE
    if kiezpresort:
        np.fill_diagonal(distance_matrix, 99999)
        stuff = np.argpartition(distance_matrix, kiezpresort, axis=1)[:,kiezpresort:]
        # we dont want to set the value we want to add, how would we do that?
        distance_matrix[np.arange(n)[:, None], stuff] += 99999
        # np.put_along_axis(distance_matrix, stuff, 99999, axis=1)

    # Apply scaling
    for i in range(n):
        for j in range(n):
            v = distance_matrix[i,j]
            distance_matrix[i,j]  =  f(v,knn_good[i],knn_bad[j])

    return distance_matrix

def csls_(v,i,j):
    return v*2 -i -j
def ls(v,i,j):
    return 1- np.exp(- v**2/(i*j) )
def nicdm(v,i,j):
    return v /  np.sqrt(i*j)
def ka(v,i,j):
    return v / i +  v/j

def format_dist_ind(matrix,k, rmdiag = True):
    matrix2 = matrix.copy()
    if rmdiag:
        np.fill_diagonal(matrix2, 99999)
    mask = np.argsort(matrix2, axis=1)[:,:k]
    return np.take_along_axis(matrix, mask, axis=1), mask



def transform_experiments(distance_matrix, k=10, algo = 2, kiezbug = 0, kiezpresort = 0):
    '''

    '''
    k= int(k)
    algo = int(algo)
    kiezbug = int(kiezbug)
    kiezpresort = int(kiezpresort)

    if algo == 0:
        return distance_matrix
    if algo == 1:
        return sklearn.preprocessing.normalize(distance_matrix, axis = 0)


    n = distance_matrix.shape[0]

    # Compute KNN mean distances
    startfrom = 1
    knn_indices = np.argpartition(distance_matrix, k + startfrom, axis=1)[:, :k + startfrom]
    knn_distances = np.take_along_axis(distance_matrix, knn_indices, axis=1)
    knn_distances = np.sort(knn_distances, axis=1) # Sort the k+startfrom selected distances
    knn_good = knn_distances[:, startfrom:].mean(axis=1) # Mean from startfrom

    if kiezbug:
        knn_bad =  knn_distances[:, :-1].mean(axis=1) # Mean from 0 up to k+startfrom-1
    else:
        knn_bad = knn_good

    # MAKING THINGS WORSE (as in the original function)
    if kiezpresort:
        temp_matrix = distance_matrix.copy()
        np.fill_diagonal(temp_matrix, np.inf) # Use infinity or a large value
        stuff_indices = np.argsort(temp_matrix, axis=1)[:, kiezpresort:] # Sort fully and take the large distances
        row_indices = np.arange(n)[:, None] # Column vector [0, 1, ..., n-1].T
        distance_matrix[row_indices, stuff_indices] += 99999

    # Apply scaling using broadcasting
    knn_good_col = knn_good[:, None] # Shape (n, 1)
    knn_bad_row = knn_bad[None, :]   # Shape (1, n)

    if algo == 2: # csls_ : v*2 - i - j
        distance_matrix = distance_matrix * 2 - knn_good_col - knn_bad_row
    elif algo == 3: # ls : 1 - np.exp(- v**2/(i*j) )
        # Add a small epsilon to the denominator to avoid division by zero if knn_good[i] or knn_bad[j] is zero
        epsilon = 1e-8
        denominator = (knn_good_col * knn_bad_row)
        # Handle cases where denominator might be zero or negative
        # Replace non-positive values with epsilon before taking square root or using in division
        denominator = np.where(denominator <= 0, epsilon, denominator)

        distance_matrix = 1 - np.exp(- distance_matrix**2 / denominator)
    elif algo == 4: # nicdm : v /  np.sqrt(i*j)
         # Add a small epsilon to the denominator to avoid division by zero if knn_good[i] or knn_bad[j] is zero
        epsilon = 1e-8
        denominator = (knn_good_col * knn_bad_row)
        # Handle cases where denominator might be zero or negative
        # Replace non-positive values with epsilon before taking square root or using in division
        denominator = np.where(denominator <= 0, epsilon, denominator)
        distance_matrix = distance_matrix / np.sqrt(denominator)

    elif algo == 5: # ka : v / i +  v/j
        # Add a small epsilon to denominators to avoid division by zero
        epsilon = 1e-8
        knn_good_col = np.where(knn_good_col <= 0, epsilon, knn_good_col)
        knn_bad_row = np.where(knn_bad_row <= 0, epsilon, knn_bad_row)

        distance_matrix = distance_matrix / knn_good_col + distance_matrix / knn_bad_row

    return distance_matrix
