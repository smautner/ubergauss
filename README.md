# ubergauss


kneepoint detection (using 2 gaussians or max dist to diagonal) and sklearn gmm wrapper



```python
import ubergauss as ug

# kneepoint gaussians:
>>>ug.between_gaussians([.1,.1,.2,.2,.4,.7])
4


# max dist to diagonal
>>>ug.diag_maxdist([.1,.1,.2,.2,.4,.7])
3


# trian gmm
ug.get_model(X, poolsize = -1,
                nclust_min = 4,
                nclust_max = 20,
                n_init = 30,
                covariance_type = 'tied',
                kneepoint_detection = diag_maxdist,
                **kwargs)
```



# sigma boxplot

![''](https://raw.githubusercontent.com/smautner/ubergauss/master/test/Figure_1.png)
![''](https://raw.githubusercontent.com/smautner/ubergauss/master/test/Figure_2.png)

# optimization:

blackboxBORE is the one that works best

![''](https://raw.githubusercontent.com/smautner/ubergauss/master/test/optimize.png)
