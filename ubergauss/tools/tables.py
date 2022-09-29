
from lmz import Transpose

import numpy as np

def totab(dlist):
    keys = dlist[0].keys()
    nutab =  [ [d[k] for k in keys]  for d in dlist]
    return nutab,keys



def touniquetab(tab):
    '''
    returns: keys, truncated table
    '''
    tab,k = totab(tab)

    ttab = Transpose(tab)
    allsame = [True if all([rr==row[0] for rr in row]) else False for row in ttab]
    ttab = [ t for t,s in zip(ttab,allsame) if not s]
    k = [ kk for kk,s in zip(k,allsame) if not s]

    return k, Transpose(ttab)


if __name__ == "__main__":
    tab = [{'type': 'type_1', 'prob': 2, 'x_sum': 3, 'y_sum': 5},
     {'type': 'type_2', 'prob': 3, 'x_sum': 3, 'y_sum': 6}]
    print(touniquetab(tab))

