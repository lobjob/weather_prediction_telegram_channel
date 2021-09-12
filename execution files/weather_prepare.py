import pandas as pd

def load_csv(path):
    df = pd.read_csv(
        path, header=6, sep=';', index_col=[0],
        parse_dates=[0], names=new_columns, dayfirst=True
    ).sort_index(ascending=True)
    return df

def new_time_features(df, max_temp_col, min_temp_col, press_col, hum_col):
    all_columns = [max_temp_col, min_temp_col, press_col, hum_col]
    temp_columns = [max_temp_col, min_temp_col]
    df['month'] = df.index.month
    df['day'] = df.index.day
    
    for col in all_columns:
        df['rolling_mean_'+col] = df[col].shift(7).rolling(10).mean()
        
    df['lag_7_max_temp'] = df[max_temp_col].shift(7)
    df['lag_20_max_temp'] = df[max_temp_col].shift(20)
    
    for col in temp_columns:
        df['lag_15_'+col] = df[col].shift(15)
        df['lag_180_'+col] = df[col].shift(180)
    return df

new_columns = [
    'температура', 'атм_давл_на_ст', 'атм_давл_на_ур_мор', 'изм_атм_давл', 'отн_влаж',
    'напр_ветра', 'скор_ветра', 'макс_порыв_ветра1', 'макс_порыв_ветра2', 'облач', 'тек_погода',
    'прош_погода1', 'прош_погода2', 'мин_темп', 'макс_темп', 'облака', 'колич_обл', 'выс_обл',
    'облака1', 'облака2', 'дальн_вид', 'темп_точки_росы', 'колич_осадков', 'время_накопл_осадков',
    'поверхн_почвы', 'темп_почвы', 'поверхн_почвы2', 'выс_снега', ''
]

df_1 = load_csv('Datasets/weather/weather_2005-2010.csv')
df_2 = load_csv('Datasets/weather/weather_2010-2015.csv')
df_3 = load_csv('Datasets/weather/weather_2015-2020.csv')
df_4 = load_csv('Datasets/weather/weather_last.csv')

df_1 = df_1[['температура', 'атм_давл_на_ст', 'отн_влаж']]
df_2 = df_2[['температура', 'атм_давл_на_ст', 'отн_влаж']]
df_3 = df_3[['температура', 'атм_давл_на_ст', 'отн_влаж']]
df_4 = df_4[['температура', 'атм_давл_на_ст', 'отн_влаж']]

df_all = pd.concat([df_1, df_2, df_3, df_4])
df_all_max_temp = df_all['температура'].to_frame().astype('float')
df_all_final = df_all_max_temp.resample('1D').max()
df_all_final = df_all_final.rename(columns={'температура':'max_temp'})

df_all_min_temp = df_all['температура'].to_frame().astype('float')
df_all_min_temp = df_all_min_temp.resample('1D').min()
df_all_min_temp = df_all_min_temp.rename(columns={'температура':'min_temp'})
df_all_final = df_all_final.join(df_all_min_temp, how='left')

columns_for_remake = ['атм_давл_на_ст', 'отн_влаж']
for col in columns_for_remake:
    df_all_new_column = df_all[col].to_frame().astype('float')
    df_all_new_column = df_all_new_column.resample('1D').mean().round(1)
    df_all_final = df_all_final.join(df_all_new_column, how='left')
df_all_final = df_all_final.rename(columns={'атм_давл_на_ст':'pressure', 'отн_влаж':'humidity'})

df_all_final.loc['2008-05-06'] = df_all_final.loc['2008-05-06'].fillna(
    df_all_final.loc['2008-05-04':'2008-05-08'].median())
df_all_final.loc['2012-12-16'] = df_all_final.loc['2012-12-16'].fillna(
    df_all_final.loc['2012-12-14':'2012-12-18'].median())
new_time_features(df_all_final, 'max_temp', 'min_temp', 'pressure', 'humidity')
df_all_final = df_all_final.dropna()

df_all_final.to_csv('Datasets/weather/weather.csv')