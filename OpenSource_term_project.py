import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression      
from sklearn.preprocessing import PolynomialFeatures 

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 

def read_csv():
    water_temp_df = pd.read_csv('./0201_2201_avg_water_temp.csv', encoding='cp949')
    snow_cover_df = pd.read_csv('./0201_2201_snow_time.csv', encoding='cp949') 
    pressure_df = pd.read_csv('./0201_2201_avg_press.csv', encoding='cp949')
    
    water_temp_df.drop('지점', axis = 1, inplace = True)
    water_temp_df.rename(columns = {'일시': 'date', '평균 수온(°C)':'w_temp'}, inplace=True)
    water_temp_df['date'] = pd.to_datetime(water_temp_df['date'])
    date_ , w_temp_ = list(), list()
    for i in range(0, water_temp_df.shape[0]):
        tmp = water_temp_df.loc[i]
        date_.extend([tmp['date'], tmp['date'] + timedelta(hours = 12)])
        w_temp_.extend([tmp['w_temp'], tmp['w_temp']])
    water_temp_df_mod = pd.DataFrame({'date' : date_, 'w_temp' : w_temp_})
    water_temp_df_mod.set_index('date', drop = True, inplace = True)

    snow_cover_df.drop({'지점명', '지점'}, axis = 1, inplace = True)
    snow_cover_df.rename(columns = {'일시':'date', '일 최심신적설(cm)':'snow_cover', '일 최심신적설 시각(hhmi)':'cover_time'}, inplace=True)
    snow_cover_df['date'] = pd.to_datetime(snow_cover_df['date'])
    snow_cover_df = snow_cover_df.fillna(0)
    snow_cover_, date_ = list(), list()
    for i in range(0, snow_cover_df.shape[0]):
        tmp = snow_cover_df.loc[i]
        date = tmp['date']
        if int(tmp['cover_time']) >= 1200:
            date = date + timedelta(hours = 12)
        snow_cover_.append(tmp['snow_cover'])
        date_.append(date)
    snow_cover_df_mod = pd.DataFrame({'date' : date_, 'snow_cover' : snow_cover_})
    snow_cover_df_mod.set_index('date', drop = True, inplace = True)

    pressure_df = pressure_df[pressure_df['기압(hPa)'] == 700]
    pressure_df.rename(columns = {'일시(UTC)':'date', '기압(hPa)':'기압', '고도(gpm)':'고도', '기온(°C)':'air_temp'}, inplace=True)
    pressure_df.drop(['지점', '지점명', '기압', '고도'], axis = 1, inplace = True)
    pressure_df['date'] = pd.to_datetime(pressure_df['date'])
    pressure_df.set_index('date', drop = True, inplace = True)
    
    res = pd.concat([pressure_df, water_temp_df_mod, snow_cover_df_mod], axis = 1)
    res[['air_temp', 'w_temp']].dropna()
    res['wt_minus_at'] = res['w_temp'] - res['air_temp']
    res['snow_cover'] = res['snow_cover'].fillna(-1)
    
    res['bool_snow'] = res['snow_cover'].apply(lambda x : False if x == -1 else True)

    res = res.dropna()
    print(res)
    return res
    

def crawling_badatime(y, m, d):
    response = requests.get("https://www.badatime.com/239-{0}-{1}-{2}.html".format(y, m, d))
    soup = BeautifulSoup(response.text, 'html.parser')
    tmp = soup.findAll('td', attrs = {"rowspan":"2"})
    date_weather = dict()
    date = None
    p = re.compile('[0-9]*')
    for line in tmp:
        text = line.get_text().split('\n')
        text = ' '.join(text).split()
        if text[0][0:1].isdigit():
            day = p.match(text[0]).group()
            date = datetime(int(y), int(m), int(day))
        else:
            if len(text) == 1:
                date_weather[date] = [text[0], text[0]]
            elif len(text) == 2:
                date_weather[date] = text
            else:
                raise Exception
    return dic_to_df(date_weather)


def dic_to_df(dic):
    index, data = list(), list()
    cnt = 0
    for key, val in dic.items():
        if cnt != 0 and index[cnt - 1] + timedelta(hours = 12) != key:
            while index[cnt - 1] + timedelta(days = 1) != key: 
                index.extend([index[cnt - 1] + timedelta(days = 1), index[cnt - 1] + timedelta(days = 1) + timedelta(hours = 12)])
                data.extend([0, 0])
                cnt += 2
        index.extend([key, key + timedelta(hours = 12)])
        data.extend([val[0], val[1]])
        cnt += 2
    res = pd.DataFrame({'date' : index, 'weather' : data})
    res.set_index('date', drop = True, inplace = True)
    return res


def crawling_devweather(y, m):
    response = requests.get("https://www.weather.go.kr/w/obs-climate/land/past-obs/obs-by-day.do?stn=102&yy={0}&mm={1}&obs=9".format(y, m))
    soup = BeautifulSoup(response.text, 'html.parser')
    tmp = soup.findAll('table', attrs = {"class":"table-col table-cal"})
    date, precipitation = list(), list()
    for text in tmp:
        tmp_ = text()
        print(tmp_)


def poly_reg(data):
    idx = data[data['snow_cover'] == -1].index
    data.drop(idx, inplace = True)
    print(data)
    x = data[['wt_minus_at']]
    y = data['snow_cover']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10) 
    
    poly = PolynomialFeatures(degree=2)           
    X_train_poly=poly.fit_transform(x_train)  
    
    pr = LinearRegression()   
    pr.fit(X_train_poly, y_train)

    # 학습을 마친 모형에 test data를 적용하기 위해 X_test 데이터를 2차항으로 변형
    X_test_poly = poly.fit_transform(x_test)

    y_hat_test = pr.predict(X_test_poly)
    y_comp2 = pd.DataFrame({'y':y_test, 'y_hat':y_hat_test})
    print(y_comp2)
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_train, y_train, 'o', label='Train Data')  # 데이터 분포
    ax.plot(x_test, y_hat_test, 'r+', label='Predicted Value') # 모형이 학습한 회귀선
    plt.show()
    plt.close() 
    
    '''
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)   
    y_hat = knn.predict(x_test)

    print(y_hat[0:10])
    print(y_test.values[0:10])
    '''


def knn(data):
    X=data[['wt_minus_at']]  #독립 변수 X
    y=data['bool_snow']                      #종속 변수 Y

    X = preprocessing.StandardScaler().fit(X).transform(X)

    print(X)
    print('\n')
    


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10) 
    knn = KNeighborsClassifier(n_neighbors=5)

    # train data를 가지고 모형 학습
    knn.fit(X_train, y_train)   

    # test data를 가지고 y_hat을 예측 (분류) 
    y_hat = knn.predict(X_test)

    print(y_hat[0:10])
    print(y_test.values[0:10])

    # 모형 성능 평가 - Confusion Matrix 계산
    from sklearn import metrics 
    knn_matrix = metrics.confusion_matrix(y_test, y_hat)  
    print(knn_matrix)

    # 모형 성능 평가 - 평가지표 계산
    knn_report = metrics.classification_report(y_test, y_hat)            
    print(knn_report)

def main():
    res = read_csv()
    knn(res)
    #res.plot(kind='scatter', x = 'wt_minus_at', y = 'snow_cover')
    

#plt.show()
#print(crawling_badatime('2014', '12', '01').head(30))
#crawling_devweather('2014', '12')

#res.plot(kind='scatter', x = 'wt_minus_at', y = 'snow_cover', c = 'coral')
#sns.regplot(x = 'wt_minus_at', y = 'snow_cover', data = res)
#res.plot(kind='line', x = 'wt_minus_at', y = 'snow_cover')

#plt.show()

main()