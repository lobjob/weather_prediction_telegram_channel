# Проект: Канал в Телеграме с предсказаниями погоды в Москве.

Создал канал/бот в Телеграме, в который еженедельно постится предсказвния погоды от модели машинного обучения. Модель обучается на наборе исторических данных наблюдений за погодой в Москве за последние 15 лет.

Алгоритм действий:
- модель, раз в неделю получает свежие данные о погоде;
- обучается и проходит валидацию и тестирование;
- каждое воскресенье, присылает в канал прогноз погоды на следующую неделю.

Задачи:
- Обработать данные о погоде, понять что в них представляет ценность.
- Понять, что и как можно автоматизировать при обработке новых данных.
- Задействовать Телеграм-бота для работы сервиса.
- Обучить и протестировать модель.
- Добиться хорошей точности прогноза.
- Сделать хороший вывод информации в Телеграм-канал.

Данные получены с сайта:
[rp5.ru](https://rp5.ru/Погода_в_мире"https://rp5.ru/Погода_в_мире")

Текущие результаты:
- данные изучены, обработаны и подготовлены к моделированию;
- модель успешно обучается, проходит валидацию, тест и проверку на адекватность;
- модель делает прогноз и выводит его в [телеграм-канал](https://t.me/weather_cat"https://t.me/weather_cat").

Что планирую делать дальше:
- добавить прогноз минимальной суточной температуры (пока прогнозируется только максимальная);
- улучшить качество модели с помощью добавления дополнительных признаков (давление, влажность и т.п.);
- автоматизировать процесс сбора данных, подготовки данных и прогноза.
