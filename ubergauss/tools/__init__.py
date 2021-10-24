from multiprocessing import Pool
_func = None
def worker_init(func):
  global _func
  _func = func
def worker(x):
  return _func(x)
def xmap(func, iterable, processes=None):
  with Pool(processes, initializer=worker_init, initargs=(func,)) as p:
    return p.map(worker, iterable)

