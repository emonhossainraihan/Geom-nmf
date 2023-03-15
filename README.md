## Create a virtual env

```shell
> pip install virtualenv
> virtualenv env
> .\env\Script\activate
(env) > .\env\Script\python.exe -m pip install -r requirements.txt
```

OR

```shell
pip install numpy pandas scipy
```

How to active your env from cmd:

```
& <your_dir>/NMF/env/Scripts/Activate.ps1
```

Now watch `image_test.ipynb` file.

### Data Information

`CBCL.mat` file contains 2429 images (columns), each image (column) contain a 19x19 matrix (361).

### Question

- [x] [Multiplicative update Rule in PyTorch](https://stackoverflow.com/q/75742628/9138425)
- [x] Find operator [1](https://math.stackexchange.com/q/4658970/871843)
- [ ] Regularization + Initialization

### TODOs

- [ ] Use PyTorch to get [Multiplicative update Rule](https://stats.stackexchange.com/a/352921/312701)
- [ ] Add custom Regularization + Initialization
- [ ] Add Geometric operator like [this](https://sci-hub.ru/https://www.worldscientific.com/doi/epdf/10.1142/S021969131940006X)
