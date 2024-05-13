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



def get_columns_fromarg(X,log_columns):
    if callable(log_columns):
        return log_columns(X)
    elif hasattr(log_columns, "__len__"):
        return log_columns
    else:
        return X.columns

class CombinedRegressor:
    def __init__(self,n_neighbors=7,log_columns=None,start_with_log=False,threshold=1000):
        """
            n_neighbors : Number of neighbors in knn implementation
            log_columns=None : Could obe a function taking the X dataset return a list of columns, or a list of columns, on which the log must be applied
            start_with_log=False : Whether or not the first tested regressor is the log one or not
            threshold=1000 : Where to transition from one regressor to the other
        """
        self.threshold=threshold
        self.start_with_log=start_with_log
        self.n_neighbors=n_neighbors
        self.log_columns=log_columns
        self.non_log_regressor = KNeighborsRegressor(n_neighbors)
        self.log_regressor = KNeighborsRegressor(n_neighbors)

    def fit(self,X_train, y_train):
        self.non_log_regressor.fit(X_train, y_train)
        X_log_train = X_train.copy()
        y_train_log = np.array(y_train)
        y_train_log = np.log(y_train_log+1)
        for col in get_columns_fromarg(X_log_train,self.log_columns):
            X_log_train[col] = np.log(X_log_train[col]+1)
        self.log_regressor.fit(X_log_train,y_train_log)

    def predict(self,X_test):
        X_log_test = X_test.copy()
        for col in get_columns_fromarg(X_log_test,self.log_columns):
            X_log_test[col] = np.log(X_log_test[col]+1)
        if self.start_with_log:
            y_pred_1 = self.log_regressor.predict(X_log_test)
            y_pred_1 = np.exp(y_pred_1)-1
            y_pred_2 = self.non_log_regressor.predict(X_test)
        else:
            y_pred_1 = self.non_log_regressor.predict(X_test)
            y_pred_2 = self.log_regressor.predict(X_log_test)
            y_pred_2 = np.exp(y_pred_2)-1
        y_pred = np.zeros(len(y_pred_2))
        for i in range(len(y_pred)):
            if(y_pred_1[i]>self.threshold and self.start_with_log):
                y_pred[i]=y_pred_2[i]
            elif(y_pred_1[i]<=self.threshold and self.start_with_log):
                y_pred[i]=y_pred_1[i]
            elif(y_pred_1[i]<=self.threshold and not self.start_with_log):
                 y_pred[i]=y_pred_2[i]
            else:
                y_pred[i]=y_pred_1[i]
        return y_pred
    


def fit_and_predict(X_train, y_train, X_test, y_test, regressor, verbose=False):
    assert isinstance(regressor, CombinedRegressor)
    print("fitting...")
    regressor.fit(X_train, y_train)


    print("predicting...")
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
    print("- test_combined(n_neighbors=7,start_with_log=False,threshold=1000)")
    print("- evaluate()")
    print("- accuracy(threshold)")
    print("- normalize()")
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

    def get_columns(X):
        cols = []
        for i in range(1,len(X.columns)+1):
            if 'DAY_'+str(i) in X.columns:
                cols.append('DAY_'+str(i))
        return cols


    def test_combined(n_neighbors=7,start_with_log=False,threshold=1500):
        global y_pred
        global model_used
        global X_train
        global y_train
        global X_test
        global y_test
        model_used = 0
        regressor = CombinedRegressor(n_neighbors=n_neighbors,start_with_log=start_with_log,threshold=threshold)
        y_pred = fit_and_predict(X_train, y_train, X_test, y_test, regressor)

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
                overall_mean += y_test_arr[i]
                if(abs(y_test_arr[i]-y_pred[i])/y_test_arr[i]<threshold):
                    accuracy += 1
                    mean_value += y_test_arr[i]
        mean_value /= accuracy
        overall_mean /= c
        accuracy /= c
        print("The validation accuracy at a threshold of",threshold,"is :",accuracy)
        print("The mean of the positive tests is",round(mean_value,1),"while the overall mean is",round(overall_mean,1))



    def evaluate():
        global y_pred
        global count
        count = -1
        print("Performance for combined_model : ")
        evaluate_performance(y_test, y_pred)
        print('\nUse arrow keys up and down to navigate, and Q to exit\n')
        try:    
            from pynput.keyboard import Key, Listener
        except Exception:
            print('Execute : `pip install pynput` to get better control\n')
            return
            
        y_test_array = np.array(y_test)
        def handlePress(key):
            global count
            if key==Key.down:
                count+=1
                if count>=len(y_test):
                    count -= len(y_test)
                print('\b\b\b\b\033[1A                              \r',str(count+1)+'/'+str(len(y_test))+" : ",y_test_array[count],' -> ',y_pred[count],sep='',end='\n')
            elif key==Key.up:
                count-=1
                if count<0:
                    count += len(y_test)
                print('\b\b\b\b\033[1A                              \r',str(count+1)+'/'+str(len(y_test))+" : ",y_test_array[count],' -> ',y_pred[count],sep='',end='\n')
                
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