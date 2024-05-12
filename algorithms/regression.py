import os
import sys
from pathlib import Path
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression




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
    print('\n')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

def help():
    print("Type in some of the following : ")
    print("- test_linear()")
    print("- test_knn(n_neighbors)")
    print("- evaluate()")
    print("- accuracy()")
    print("- normalize()")
    print("- set_log()")
    print("- remove_day(day)")
    print("")




if __name__ == "__main__":
    import_folder = str(Path(__file__).parent.resolve())+"/../datasets/"
    import_path = "dataset.csv"

    regression_column = "NBRE_VALIDATION"

    add_weekdays=1
    is_log=0


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
        add_weekdays = int(sys.argv[-2])
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

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=20000, random_state=4)



    y_pred = np.array(y_test)

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

    def set_log():
        global is_log
        global y_test
        global y_train
        global X_test
        global X_train
        if(not is_log):
            is_log=True
            y_train = np.log(y_train.copy()+1)
            y_test = np.log(y_test.copy()+1)
            for i in range(1,8):
                try:
                    X_test['DAY_'+str(i)] = np.log(X_test['DAY_'+str(i)]+1)
                    X_train['DAY_'+str(i)] = np.log(X_train['DAY_'+str(i)]+1)
                finally:
                    ""
            print("Done!")
        else:
            print("Already Done!")
        
    def remove_day(day=None):
        global X_test
        global X_train
        if day==None:
            print("You need to specify which day(int), here are the available days :",[col for col in X_train.columns if "DAY_" in col],"\n")
        elif isinstance(day, int):
            try:
                X_test = X_test.drop(columns=['DAY_'+str(day)])
                X_train = X_train.drop(columns=['DAY_'+str(day)])
            except:
                print("Error")
        else:
            print(day,"is not an int")
            

    count = -1

    def get_value(x):
        global is_log
        if(is_log):
            return round(np.exp(x)-1,1)
        else:
            return round(x,1)
        
    def accuracy(threshold=0.1):
        global y_pred
        global y_test
        accuracy = 0
        c = 0
        y_test_arr = np.array(y_test)
        mean_value = 0
        overall_mean = 0
        for i in range(len(y_pred)):
            if y_test_arr[i]!=0:
                c += 1
                overall_mean += get_value(y_test_arr[i])
                if(abs(get_value(y_test_arr[i])-get_value(y_pred[i]))/get_value(y_test_arr[i])<threshold):
                    accuracy += 1
                    mean_value += get_value(y_test_arr[i])
        mean_value /= accuracy
        overall_mean /= c
        accuracy /= c
        print("The validation accuracy at a threshold of",threshold,"is :",accuracy)
        print("The mean of the positive tests is",round(mean_value,1),"while the overall mean is",round(overall_mean,1))

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
                print('\b\b\b\b\033[1A                              \r',str(count+1)+'/'+str(len(y_test))+" : ",get_value(y_test_array[count]),' -> ',get_value(y_pred[count]),sep='',end='\n')
            elif key==Key.up:
                count-=1
                if count<0:
                    count += len(y_test)
                print('\b\b\b\b\033[1A                              \r',str(count+1)+'/'+str(len(y_test))+" : ",get_value(y_test_array[count]),' -> ',get_value(y_pred[count]),sep='',end='\n')
                
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