### Create a virtual env

```shell
> pip install virtualenv
> virtualenv env
> .\env\Script\activate
(env) > .\env\Script\python.exe -m pip install -r requirements.txt
```

How to active your env:

```
& <your_dir>/NMF/env/Scripts/Activate.ps1
```

OR install necessary packages using pip,

```shell
pip install numpy pandas scipy
```

Now watch `image_test.ipynb` file.

### Data Information

`CBCL.mat` file contains 2429 images (columns), each image (column) contain a 19x19 matrix (361).

### Question

- [x] [Multiplicative update Rule in PyTorch](https://stackoverflow.com/q/75742628/9138425)
- [x] Find operator [1](https://math.stackexchange.com/q/4658970/871843), ...
- [ ] Regularization + Initialization from [1](https://sci-hub.ru/https://www.sciencedirect.com/science/article/abs/pii/S0031320307004359) $\rightarrow$ [sklearn.decomposition](https://github.com/scikit-learn/scikit-learn/blob/530dfc9631b2135412a048b5ec7cf01d155b6067/sklearn/decomposition/_nmf.py#L273), ...

### TODOs

- [ ] Make sure optimized codebase from [`sklearn.decomposition`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_nmf.py), [gh pr checkout 8765]()
- [ ] Use PyTorch to get [Multiplicative update Rule](https://stats.stackexchange.com/a/352921/312701)
- [ ] Add custom Regularization + Initialization
  - [x] [image data](https://stackoverflow.com/questions/33610825/normalization-in-image-processing)
  - [x] Use max/min singular value incorporate with update rules
- [ ] Add Geometric operator from [1](https://sci-hub.ru/https://www.worldscientific.com/doi/epdf/10.1142/S021969131940006X), ...
- [ ] Add [Manifold learning](https://github.com/drewwilimitis/Manifold-Learning) type Regularization from [1](https://github.com/scikit-learn/scikit-learn/tree/0a3e585d5651af80430834c2a4008ac96ce04a21/sklearn/manifold), ...
- [ ] Low-rank matrix-factorization with better rank-proxies (trace-norm or max-norm)

### Topics to explore

- [sparseness](https://math.stackexchange.com/questions/4212759/differentiable-sparsity-measure)
- [Constrained optimization](https://www.youtube.com/watch?v=lvzH88DDaow&ab_channel=XiaojingYe)
- [Nonnegative Matrix Factorizations for Clustering, Haesun Park, Georgia Institute of Technology](https://www.youtube.com/watch?v=EKvh4ANUHWM&ab_channel=MMDSFoundation)

### Are you a coder? then check this out!!!

If are a coder, and love to deep dive into code then check [demystify.md](https://github.com/emonhossainraihan/Geom-nmf/blob/main/development/demystify.md)

First nmf in [sklearn.decomposition](https://github.com/scikit-learn/scikit-learn/blob/a2a4257bc6e793faf6867cfe781cdfad7e5a7b41/sklearn/decomposition/nmf.py)
