import pandas as pd

def read_csv():
    water_temp_df = pd.read_csv('./1411_1501_avg_water_temp_mod.csv', encoding='cp949')
    pressure_df = pd.read_csv('./1411_1501_avg_press_mod.csv', encoding='cp949')
    
    water_temp_df.drop('지점', axis = 1, inplace = True)
    water_temp_df.rename(columns = {'일시': 'date', '평균 수온(°C)':'w_temp'}, inplace=True)

    pressure_df = pressure_df[pressure_df['기압(hPa)'] == 850]
    pressure_df.rename(columns = {'일시(UTC)':'date', '기압(hPa)':'기압', '고도(gpm)':'고도', '기온(°C)':'air_temp'}, inplace=True)
    pressure_df.reset_index(drop = True, inplace = True)
    pressure_df.drop(['지점', '지점명', '기압', '고도'], axis = 1, inplace = True)
    pressure_df['date'] = pd.to_datetime(pressure_df['date'])

    tmp_w_temp = list()
    tmp_w_minus_a = list()
    for i in range(0, pressure_df.shape[0]):
        tmp = float(water_temp_df.loc[i//2]['w_temp'])
        tmp_w_temp.append(tmp)
        if pressure_df.loc[i]['air_temp'] != None:
            tmp_w_minus_a.append(float(pressure_df.loc[i]['air_temp'] - tmp))       
    pressure_df['w_temp'] = tmp_w_temp
    pressure_df['wt_minus_at'] = tmp_w_minus_a
   
    for i in range(0, pressure_df.shape[0]):
        row = pressure_df.loc[i]
        if row['wt_minus_at'] <= -20:
            print(row['date'].date(), row['wt_minus_at'])

read_csv()