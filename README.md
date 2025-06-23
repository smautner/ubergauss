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

### embedding of distance matrix

### collection of small tools

### sigma boxplot

![''](https://raw.githubusercontent.com/smautner/ubergauss/master/test/Figure_1.png)
![''](https://raw.githubusercontent.com/smautner/ubergauss/master/test/Figure_2.png)

### optimization

blackboxBORE is the one that works best

![''](https://raw.githubusercontent.com/smautner/ubergauss/master/test/optimize.png)


```
 def test_ga():
     ut.nuke()
     def example_function(data, x=None, y=None, some_boolean=None,**kwargs):
         score_from_x = - (x - 0.5)**2  # Max at x=0.5
         score_from_y = - (y - 10)**2 / 100.0 # Max at y=10
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
```
