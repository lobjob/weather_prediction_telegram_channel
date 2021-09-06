import pandas as pd

def load_csv(path):
    df = pd.read_csv(
        path, header=6, sep=';', index_col=[0],
        parse_dates=[0], names=new_columns, dayfirst=True
    ).sort_index(ascending=True)
    return df

def new_time_features(df, key_column):
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['rolling_mean'] = df[key_column].shift(7).rolling(7).mean()
    df['lag_7'] = df[key_column].shift(7)
    df['lag_15'] = df[key_column].shift(15)
    df['lag_30'] = df[key_column].shift(30)
    df['lag_90'] = df[key_column].shift(90)
    df['lag_180'] = df[key_column].shift(180)
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

df_1 = df_1['температура'].to_frame().astype('float')
df_2 = df_2['температура'].to_frame().astype('float')
df_3 = df_3['температура'].to_frame().astype('float')
df_4 = df_4['температура'].to_frame().astype('float')

df_all = pd.concat([df_1, df_2, df_3, df_4])
df_all = df_all.resample('1D').max()
df_all = df_all.rename(columns={'температура':'temperature'})

df_all.loc['2008-05-06'] = df_all.loc['2008-05-06'].fillna(df_all.loc['2008-05-04':'2008-05-08'].mean())
df_all.loc['2012-12-16'] = df_all.loc['2012-12-16'].fillna(df_all.loc['2012-12-14':'2012-12-18'].mean())

new_time_features(df_all, 'temperature')
df_all = df_all.dropna()

df_all.to_csv('Datasets/weather/weather.csv')