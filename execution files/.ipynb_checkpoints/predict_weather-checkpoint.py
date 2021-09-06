import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
from catboost import CatBoostRegressor
from notifiers import get_notifier

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

def search_best_model(model, model_name, params, X, y):
    clf = GridSearchCV(model, params, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
    clf.fit(X, y)
    score = -1 * clf.best_score_
    return score, clf.best_estimator_

def test_and_score(model):
    model.fit(X, y)
    return mean_absolute_error(y_test, model.predict(X_test))

def make_predictions(df, model):
    df_pred = df.iloc[-7:]
    X_pred = df_pred.drop('temperature', axis=1)
    preds = best_cb_model.predict(X_pred)
    
    columns_to_drop = ['month', 'day', 'rolling_mean', 'lag_7',
                       'lag_15', 'lag_30', 'lag_90', 'lag_180']
    X_pred = X_pred.copy()
    X_pred['День недели'] = X_pred.index.day_name()
    X_pred['Температура'] = preds
    X_pred['Температура'] = X_pred['Температура'].round(1)
    X_pred = X_pred.drop(columns_to_drop, axis=1)
    
    days_week_rus = {
        'Monday':'Понедельник', 'Tuesday':'Вторник', 'Wednesday':'Среда',
        'Thursday':'Четверг', 'Friday':'Пятница',
        'Saturday':'Суббота', 'Sunday':'Воскресенье'
    }
    X_pred['День недели'] = X_pred['День недели'].replace(days_week_rus)
    return X_pred

def telegram_notifier_to_channel(
    token='<bot_token>',
    chat_id='@<chanel_name>'
):
    def f(text):
        telegram = get_notifier('telegram')
        telegram.notify(
            message=text,
            token=token,
            chat_id=chat_id
        )
    return f

def telegram_notifier_to_bot(
    token='<bot_token>',
    chat_id='<chat_with_bot_id>'
):
    def f(text):
        telegram = get_notifier('telegram')
        telegram.notify(
            message=text,
            token=token,
            chat_id=chat_id
        )
    return f

df_all = pd.read_csv('Datasets/weather/weather.csv', index_col=[0], parse_dates=[0], dayfirst=True)

X = df_all.drop('temperature', axis=1)
y = df_all['temperature']
X, X_test, y, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.055, shuffle=False)

best_cb_score, best_cb_model = search_best_model(
    model=CatBoostRegressor(),
    model_name='CatBoost',
    params={'iterations': range(1000, 3100, 100), 'verbose': [0],
            'random_state': [555], 'loss_function': ['MAE']},
    X=X, y=y
)

send_bot = telegram_notifier_to_bot()
send_bot('Я обучился! Муррр!')

send_bot(
    'Мои успехи:\nValid MAE: {:.2f}\nTest MAE: {:.2f}'
    .format(best_cb_score, test_and_score(best_cb_model))
)

X = df_all.drop('temperature', axis=1)
y = df_all['temperature']
best_cb_model.fit(X, y)

next_7_dates = pd.Series(pd.date_range(df_all.index[-1], periods=7, freq='D').shift().normalize())
df_all_next = pd.DataFrame(index=next_7_dates, columns=df_all.columns)
df_all = pd.concat([df_all, df_all_next])
df_all = new_time_features(df_all, 'temperature')

predictions_temperature = make_predictions(df_all, best_cb_model)
predictions_text = ('Предсказания на неделю:'
                    '\nc "{}" по "{}"\n'
                    '\nТемпература | День недели'
                    '\n{}            {}'
                    '\n{}            {}'
                    '\n{}            {}'
                    '\n{}            {}'
                    '\n{}            {}'
                    '\n{}            {}'
                    '\n{}            {}'
                    .format(predictions_temperature.index[0].date(),
                            predictions_temperature.index[6].date(),
                            predictions_temperature.iloc[0][1],
                            predictions_temperature.iloc[0][0],
                            predictions_temperature.iloc[1][1],
                            predictions_temperature.iloc[1][0],
                            predictions_temperature.iloc[2][1],
                            predictions_temperature.iloc[2][0],
                            predictions_temperature.iloc[3][1],
                            predictions_temperature.iloc[3][0],
                            predictions_temperature.iloc[4][1],
                            predictions_temperature.iloc[4][0],
                            predictions_temperature.iloc[5][1],
                            predictions_temperature.iloc[5][0],
                            predictions_temperature.iloc[6][1],
                            predictions_temperature.iloc[6][0])
                   )
send_channel = telegram_notifier_to_channel()
send_channel(predictions_text)