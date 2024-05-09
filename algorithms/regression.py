import os
import sys
from pathlib import Path
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression




def normalize_manual(train_data, test_data, method='mean_std'):
    """
    Normalizes all the features by linear transformation *except* for the target regression column
    specified as `col_regr`.
    Two normalization methods are implemented:
      -- `mean_std` shifts by the mean and divides by the standard deviation
      -- `maxmin` shifts by the min and divides by the difference between max and min
      *Note*: mean/std/max/min are computed on the training data
    The function returns a pair normalized_train, normalized_test. For example,
    if you had `train` and `test` pandas DataFrames with the regression col stored in column `Col`, you can do

        train_norm, test_norm = normalize(train, test, 'Col')

    to get the normalized `train_norm` and `test_norm`.
    """
    # removing the class column so that it is not scaled
    no_class_train = train_data
    no_class_test = test_data

    # scaling
    normalized_train, normalized_test = None, None
    if method == 'mean_std':
        normalized_train = (no_class_train - no_class_train.mean()) / no_class_train.std()
        normalized_test = (no_class_test - no_class_train.mean()) / no_class_train.std()
    elif method == 'maxmin':
        normalized_train = (no_class_train - no_class_train.min()) / (no_class_train.max() - no_class_train.min())
        normalized_test = (no_class_test - no_class_train.min()) / (no_class_train.max() - no_class_train.min())
    else:
        raise f"Unknown method {method}"

    # gluing back the class column and returning
    return normalized_train, normalized_test





import_folder = str(Path.cwd())+"/../datasets/"
import_path = "dataset.csv"

regression_column = "NBRE_VALIDATION"

if(len(sys.argv)==1):
    print("")
    print("No argument specified, here is the template : ")
    print("\tpy opening.py <regression_column=NBRE_VALIDATION> <dataset=dataset.csv>\n")
    continue_str = input("Press enter to continue, type in 0 to exit ")
    if(continue_str=="0"):
        exit()
    print("")
elif(len(sys.argv)==2):
    import_path = sys.argv[-1]
elif(len(sys.argv)==3):
    import_path = sys.argv[-1]
    regression_column = sys.argv[-2]


def summary(dataset):
    print(f'Shape of the data {dataset.shape}')
    print(dataset.head(5))
    print(dataset.describe())
    print('\n\n')

def fit_and_predict(X_train, y_train, X_test, y_test, regressor, verbose=False):
    assert isinstance(regressor, LinearRegression) or isinstance(regressor, KNeighborsRegressor)
    regressor.fit(X_train, y_train)

    if isinstance(regressor, LinearRegression):
        print(f'\tintercept = {regressor.intercept_}')
        print(f'\tcoefficient = {regressor.coef_}')

    y_pred = regressor.predict(X_test)
    if verbose:
        for a, b in zip(y_test, y_pred):
            print(f'  true value: {a} \t predicted value: {b}')
    return y_pred




def evaluate_performance(y_test, y_pred):
    print('\n\n')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('\n')

def help():
    print("Type in some of the following : ")
    print("- test_linear()")
    print("- test_knn(n_neighbors)")
    print("- evaluate()")
    print("- normalize()")
    print("")

data = pd.read_csv(import_folder+import_path)

if "LIBELLE_LIGNE" in data.columns:
    data = data.drop(columns=["LIBELLE_LIGNE"])
y = data[regression_column]
x = data.drop(columns=[regression_column])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.03, random_state=4)

y_pred = y_test

models = ["Linear","Knn"]
model_used = 0

def test_linear():
    global y_pred
    global model_used
    global X_train
    global y_train
    global X_test
    global y_test
    model_used = 0
    regressor = LinearRegression()
    y_pred = fit_and_predict(X_train, y_train, X_test, y_test, regressor)

def test_knn(n_neighbors=5):
    global y_pred
    global model_used
    global X_train
    global y_train
    global X_test
    global y_test
    model_used = 1
    regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
    y_pred = fit_and_predict(X_train, y_train, X_test, y_test, regressor)

def evaluate():
    global y_pred
    print("Performance for "+models[model_used]+"model : ")
    evaluate_performance(y_test, y_pred)

def normalize():
    global X_train
    global X_test
    resp = input("This operation cannot be undone in this instance. \nChoose either 'mean_std' or 'maxmin' : ")
    if(resp==""):
        resp="mean_std"
    elif resp!="":
        print("Nothing Done! \n")
        return
    X_train, X_test = normalize_manual(X_train,X_test,resp)
    


if __name__ == "__main__":
    summary(X_train)
    summary(y_train)
    print("Type help() for help.\n")
    while True:
        cmd = input(">> ")
        try:
            value = eval(cmd)
            if(value!=None):
                print(value)
        except Exception as e:
            print(e)