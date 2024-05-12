import os
import sys
from pathlib import Path
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR




def normalize_data(train_data, test_data, method='mean_std'):
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






def fit_and_predict(X_train, y_train, X_test, y_test, regressor, verbose=False):
    assert isinstance(regressor, SVR)
    regressor.fit(X_train, y_train)


    y_pred = regressor.predict(X_test)
    if verbose:
        for a, b in zip(y_test, y_pred):
            print(f'  true value: {a} \t predicted value: {b}')
    return y_pred


def test_SVR(kernel='sigmoid', degree=3, C=2.0, epsilon=0.3, gamma='scale'):
    global y_pred
    global model_used
    global X_train
    global y_train
    global X_test
    global y_test
    
    try:
        kernel = kernel.lower()
        if kernel not in ['linear', 'poly', 'rbf', 'sigmoid']:
            kernel = 'rbf'
    except Exception:
        kernel = 'rbf'
    model_used = 0
    regressor = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon, degree=degree, verbose=True)
    y_pred = fit_and_predict(X_train, y_train, X_test, y_test, regressor)





def evaluate_performance(y_test, y_pred):
    print('\n')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

def help():
    print("Type in some of the following : ")
    print("- test_SVR(kernel='rbf', degree=3, C=1.0, epsilon=0.1, gamma='scale')")
    print("- evaluate()")
    print("- normalize()")
    print("")




if __name__ == "__main__":
    import_folder = str(Path(__file__).parent.resolve())+"/../datasets/"
    import_path = "dataset.csv"

    regression_column = "NBRE_VALIDATION"

    add_weekdays=1

    if(len(sys.argv)==1):
        print("")
        print("No argument specified, here is the template : ")
        print("\tpy opening.py <regression_column=NBRE_VALIDATION> <add_weekdays(int)=1> <dataset=dataset.csv>\n")
        continue_str = input("Press enter to continue, type in 0 to exit ")
        if(continue_str=="0"):
            exit()
        print("")
    elif(len(sys.argv)==2):
        import_path = sys.argv[-1]
    elif(len(sys.argv)==3):
        import_path = sys.argv[-1]
        regression_column = sys.argv[-2]
    elif(len(sys.argv)>=4):
        import_path = sys.argv[-1]
        add_weekdays = int(sys.argv[-2])
        regression_column = sys.argv[-3]


    data = pd.read_csv(import_folder+import_path)

    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    if 'DATE' in data.columns:
        data = data.drop(columns=['DATE'])
    if "LIBELLE_LIGNE" in data.columns:
        data = data.drop(columns=["LIBELLE_LIGNE"])
    if add_weekdays:
        data["IS_SUNDAY"] = 0
        data["IS_SATURDAY"] = 0
        data["IS_MONDAY"] = 0
        data["IS_TUESDAY"] = 0
        data["IS_WEDNESDAY"] = 0
        data["IS_THURSDAY"] = 0
        data.loc[data["WEEKDAY"]==5, "IS_SATURDAY"] = 1
        data.loc[data["WEEKDAY"]==6, "IS_SUNDAY"] = 1
        data.loc[data["WEEKDAY"]==0, "IS_MONDAY"] = 1
        data.loc[data["WEEKDAY"]==1, "IS_TUESDAY"] = 1
        data.loc[data["WEEKDAY"]==2, "IS_WEDNESDAY"] = 1
        data.loc[data["WEEKDAY"]==3, "IS_THURSDAY"] = 1
    if "WEEKDAY" in data.columns:
        data = data.drop(columns=["WEEKDAY"])
    y = data[regression_column]
    x = data.drop(columns=[regression_column])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.03, random_state=4)
    X_train = X_train[:20000]
    y_train = y_train[:20000]
    X_test = X_test[:5000]
    y_test = y_test[:5000]

    y_pred = np.array(y_test)

    models = ["Linear","Knn"]
    model_used = 0

    


    count = -1
    def evaluate():
        global y_pred
        global count
        count = -1
        print("Performance for "+models[model_used]+"model : ")
        evaluate_performance(y_test, y_pred)
        print('\nUse arrow keys up and down to navigate, and Q to exit\n')
        try:    
            from pynput.keyboard import Key, Listener
        except e:
            print('Execute : `pip install pynput` to get better control\n')
            return
            
        y_test_array = np.array(y_test)
        def handlePress(key):
            global count
            if key==Key.down:
                count+=1
                if count>=len(y_test):
                    count -= len(y_test)
                print('\b\b\b\b\033[1A                              \r',str(count+1)+'/'+str(len(y_test))+" : ",y_test_array[count],' -> ',round(y_pred[count],1),sep='',end='\n')
            elif key==Key.up:
                count-=1
                if count<0:
                    count += len(y_test)
                print('\b\b\b\b\033[1A                              \r',str(count+1)+'/'+str(len(y_test))+" : ",y_test_array[count],' -> ',round(y_pred[count],1),sep='',end='\n')
                
            else:
                try:
                    if key.char=='q' or key.char=='Q':
                        print('\n')
                        return False
                except Exception:
                    print('',end='')
        handlePress(Key.down)
        with Listener(on_press = handlePress) as listener:
            listener.join()
        print("")
            

    def normalize():
        global X_train
        global X_test
        resp = input("This operation cannot be undone in this instance. \nChoose either 'mean_std' or 'maxmin' : ")
        if(resp==""):
            resp="mean_std"
        elif resp=="maxmin":
            resp=resp
        elif resp!="":
            print("Nothing Done! \n")
            return
        X_train, X_test = normalize_data(X_train,X_test,resp)
        print("Normalized!\n")
        

    def summary(dataset):
        print(f'Shape of the data {dataset.shape}')
        print(dataset.head(5))
        print(dataset.describe())
        print('\n\n')

    





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