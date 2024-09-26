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
