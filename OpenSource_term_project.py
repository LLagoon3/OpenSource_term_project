import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt

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

    pressure_df = pressure_df[pressure_df['기압(hPa)'] == 850]
    pressure_df.rename(columns = {'일시(UTC)':'date', '기압(hPa)':'기압', '고도(gpm)':'고도', '기온(°C)':'air_temp'}, inplace=True)
    pressure_df.drop(['지점', '지점명', '기압', '고도'], axis = 1, inplace = True)
    pressure_df['date'] = pd.to_datetime(pressure_df['date'])
    pressure_df.set_index('date', drop = True, inplace = True)
    
    res = pd.concat([pressure_df, water_temp_df_mod, snow_cover_df_mod], axis = 1)
    res[['air_temp', 'w_temp']].dropna()
    res['wt_minus_at'] = res['w_temp'] - res['air_temp']
    print(res)
    res = res.dropna()
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
res = read_csv()
#print(crawling_badatime('2014', '12', '01').head(30))
#crawling_devweather('2014', '12')

res.plot(kind='scatter', x = 'wt_minus_at', y = 'snow_cover', c = 'coral')
#res.plot(kind='line', x = 'wt_minus_at', y = 'snow_cover')

plt.show()