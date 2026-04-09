# ubergauss


another machine learning library

## Install
```
pip install ubergauss
conda install -c conda-forge ubergauss
```

## Example

### kneepoint detection

```python
import ubergauss as ug

# kneepoint gaussians:
>>>ug.between_gaussians([.1,.1,.2,.2,.4,.7])
4
# max dist to diagonal
>>>ug.diag_maxdist([.1,.1,.2,.2,.4,.7])
3

```

### hyperparameter optimization

```
 def test_ga():
     ut.nuke()
     def example_function(data, x=None, y=None, some_boolean=None,**kwargs):
         score_from_x = - (x - 0.5)**2
         score_from_y = - (y - 10)**2 / 100.0
         score_from_bool = .1*some_boolean
         score_noise = np.random.normal(0, .1)
         return score_noise + score_from_x + score_from_y + score_from_bool

     example_space = """
     x 0.0 1.0
     y 1 20 1
     some_boolean [1, 0]
     """
     o = gatype.nutype(example_space,
                       example_function,
                       data=[[0]],
                       numsample=16)
     [o.opti() for _ in range(5)]
     o.print()
     # o.print_more()


def gridsearch(func,data_list = None, *, param_dict = False, tasks = False, taskfilter =None,
               score = 'score',mp = True,  df = True, param_string = False , timevar=f'time', **kwargs):
    '''
        # tasks
        - tasks: ive me a task list of dictionaries
        - param_dict: a dict that defines valid options {paramname: [1,2,3]}
        - param_string: either valid options param:['option1','option2']
                                or linspace param: 1 1.5 11
        - you could also use hyperopt.spaceship(string).sample() to sample tasks
    '''

```


### embedding of distance matrix
```
csrjax.embed(X,n_components=2) -> pacmap style embedder
graphumap.graphumap -> wraps umap
```

### collection of small tools

```
tools

xxmap -> multiprocessing
zehidense
(j/s)(dump/load)file -> json/sparse
spacemap/labelsToIntList
cache(fname/func)
nuke -> stupid error messages

```


### sigma boxplot

![''](https://raw.githubusercontent.com/smautner/ubergauss/master/test/Figure_1.png)
![''](https://raw.githubusercontent.com/smautner/ubergauss/master/test/Figure_2.png)

### optimization

blackboxBORE is the one that works best

![''](https://raw.githubusercontent.com/smautner/ubergauss/master/test/optimize.png)



