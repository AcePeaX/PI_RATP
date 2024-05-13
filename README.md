# Hacking into Paris Metro
This project aims to analyse RATP data to create predictive models.



## Setup
To setup the environment:

1. You need first to download the datasets. To download all of them, just execute `py utils/download_datasets.py`
1. You need then to bake some datasets. This is explained later. For example, run `py utils/export_surface.py`

## Datasets
So far, we use the datasets:
 -  [Historique des données de validation sur le réseau de surface (2015-2022)](https://www.data.gouv.fr/fr/datasets/historique-des-donnees-de-validation-sur-le-reseau-de-surface-2015-2022/)


## Files description

This project is divided into two main script categories, the `utils` and the `algorithms`.

In the `utils` folder, you will find some scripts to *bake* the data. Here are the scripts of `utils` :
- `download_datasets.py` : A script to easily download all the datasets files.
- `export_surface.py` : A script to bake the surface dataset. It mainly regroups all the files, combines the data to create the structure used for regression and other algorithms.
<br/>

In the `algorithms` folder, there are the **algorithms** used for learning and fitting the models.
- `regression.py` provides an interactive menu to test and run both **linear** and **knn** regression.



The file `plot_result.py` is just for plotting some results that we've got.
