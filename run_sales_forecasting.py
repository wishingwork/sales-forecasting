## Data source: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data?select=stores.csv
## Algorithm: https://www.kaggle.com/code/aslanahmedov/walmart-sales-forecasting/notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

def data_preprocess():
    data = pd.read_csv('../input/store-sales-time-series-forecasting/transactions.csv')
    stores = pd.read_csv('../input/store-sales-time-series-forecasting/stores.csv')
    # features = pd.read_csv('../input/store-sales-time-series-forecasting/oil.csv')

    # filling missing values
    # features['dcoilwtico'].fillna(features['dcoilwtico'].median(),inplace=True)    
    data = pd.merge(data,stores,on='store_nbr',how='left')
    # data = pd.merge(data,features,on=['date'],how='left')

    data['date'] = pd.to_datetime(data['date'],errors='coerce')
    data.sort_values(by=['date'],inplace=True)
    data.set_index(data.date, inplace=True)

    data['Year'] = data['date'].dt.year
    data['Month'] = data['date'].dt.month
    data['Week'] = data['date'].dt.isocalendar().week
    # print(data)

    #Outlier Detection and Abnormalities
    agg_data = data.groupby(['city', 'state']).transactions.agg(['max', 'min', 'mean', 'median', 'std']).reset_index()

    store_data = pd.merge(left=data,right=agg_data,on=['city', 'state'],how ='left')
    store_data.dropna(inplace=True)

    data = store_data.copy()
    # print(data)

    ## Use date for index
    data['date'] = pd.to_datetime(data['date'],errors='coerce')
    data.sort_values(by=['date'],inplace=True)
    data.set_index(data.date, inplace=True)

    cat_col = ['store_nbr','type']
    data.drop(columns=cat_col,inplace=True)
    data.drop(columns=['city', 'state', 'date'],inplace=True)

    return data

def normalize_data(data):
    num_col = ['transactions','cluster','max','min','mean','median','std']
    minmax_scale = MinMaxScaler(feature_range=(0, 1))
    def normalization(df,col):
        for i in col:
            arr = df[i]
            arr = np.array(arr)
            df[i] = minmax_scale.fit_transform(arr.reshape(len(arr),1))
        return df
    nor_data = normalization(data.copy(),num_col)
    # print(data)
    return nor_data

def get_feature_list(data):
    feature_col = data.columns.difference(['transactions'])
    radm_clf = RandomForestRegressor(oob_score=True,n_estimators=23)
    radm_clf.fit(data[feature_col], data['transactions'])
    indices = np.argsort(radm_clf.feature_importances_)[::-1]
    feature_rank = pd.DataFrame(columns = ['rank', 'feature', 'importance'])

    for f in range(data[feature_col].shape[1]):
        feature_rank.loc[f] = [f+1,
                            data[feature_col].columns[indices[f]],
                            radm_clf.feature_importances_[indices[f]]]
    x=feature_rank.loc[0:25,['feature']]
    x=x['feature'].tolist()
    print(x)
    return x

def separate_training_testing_dataset(data, nor_data):
    x = get_feature_list(nor_data)
    X = data[x]
    Y = data['transactions']
    data = pd.concat([X,Y],axis=1)
    X = data.drop(['transactions'],axis=1)
    Y = data.transactions
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.20, shuffle=False)
    return X_train,X_test,y_train,y_test

def train_for_prediction(X_train,X_test,y_train,y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    linear_regression_accuracy = lr.score(X_test,y_test)*100
    print("Linear Regressor Accuracy - ",linear_regression_accuracy)
    y_pred = lr.predict(X_test)
    return y_pred, lr

def wmae_test(test, pred): # WMAE for test 
    error = np.sum( np.abs(test - pred), axis=0) 
    return error

def print_evaluation(y_test, y_pred):
    print("MAE" , metrics.mean_absolute_error(y_test, y_pred))
    print("MSE" , metrics.mean_squared_error(y_test, y_pred))
    print("RMSE" , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("R2" , metrics.explained_variance_score(y_test, y_pred))
    
    predict_data = pd.DataFrame(y_pred, index=X_test.index)
    print(np.sum(wmae_test(y_test.resample('W').mean().values, predict_data.resample('W').mean().values)))

def plot_predict_timeline(y_train, y_test, predict_data):
    predict_data = pd.DataFrame(y_pred, index=X_test.index)

    plt.figure(figsize=(15,8))
    plt.title('Comparison between actual and predicted values',fontsize=16)

    plt.plot(y_test.resample('W').mean(), label="real_values", linewidth=1.0,color='yellow')
    plt.plot(predict_data.resample('W').mean(), label="prediction", linewidth=1.0,color='black')
    plt.plot(y_train.resample('W').mean(), label="train", linewidth=1.0,color='purple')

    plt.legend(loc="best")
    plt.show()

data = data_preprocess()
nor_data = normalize_data(data)

X_train,X_test,y_train,y_test = separate_training_testing_dataset(nor_data, nor_data)

y_pred, lr = train_for_prediction(X_train,X_test,y_train,y_test)

print_evaluation(y_test, y_pred)
plot_predict_timeline(y_train, y_test, y_pred)
