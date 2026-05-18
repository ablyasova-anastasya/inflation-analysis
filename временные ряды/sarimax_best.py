import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")


# загрузка одного временного ряда из excel
def load_series(path, value_name):
    df = pd.read_excel(path, parse_dates=['date'])
    df = df[['date', df.columns[1]]]
    df.columns = ['date', value_name]
    df['date'] = pd.to_datetime(df['date'])
    return df


# проверка стационарности ряда тестами adf и kpss
def check_stationarity(series, name):
    series = series.dropna()

    print(f"\nРяд: {name}")

    # тест Дики-Фуллера
    adf_result = adfuller(series)
    print("\nADF (Дики-Фуллер)")
    print(f"статистика = {adf_result[0]:.4f}")
    print(f"p-value    = {adf_result[1]:.4f}")
    print("вывод:", "стационарен" if adf_result[1] < 0.05 else "нестационарен")

    # тест KPSS
    kpss_result = kpss(series, regression='c', nlags='auto')
    print("\nKPSS")
    print(f"статистика = {kpss_result[0]:.4f}")
    print(f"p-value    = {kpss_result[1]:.4f}")
    print("вывод:", "стационарен" if kpss_result[1] > 0.05 else "нестационарен")


# путь к файлам
base_path = "/Users/nastya/Documents/диплом/временные ряды/"

# загрузка рядов
ipc = load_series(base_path + "ipc.xlsx", "ipc")
ipp = load_series(base_path + "ipp.xlsx", "ipp")
dollar = load_series(base_path + "dollar.xlsx", "dollar")
unemployment = load_series(base_path + "unemployment.xlsx", "unemployment")
salary = load_series(base_path + "salary.xlsx", "salary")
trading = load_series(base_path + "trading.xlsx", "trading")

# объединение всех рядов по дате
df = ipc.merge(ipp, on='date')
df = df.merge(trading, on='date')
df = df.merge(unemployment, on='date')
df = df.merge(salary, on='date')
df = df.merge(dollar, on='date')

df = df.set_index('date').asfreq('MS')

# список экзогенных факторов
exog_cols = ["ipp", "trading", "unemployment", "salary", "dollar"]

# параметры
test_size = 12
forecast_steps = 6

# проверка стационарности исходных рядов
print("\nПроверка до дифференцирования")
for col in df.columns:
    check_stationarity(df[col], col)

# создаем лаги экзогенных факторов
df_lagged = df.copy()

for col in exog_cols:
    df_lagged[col] = df_lagged[col].shift(forecast_steps)

# удаляем строки с пропусками после лагов
df_lagged = df_lagged.dropna()

# дифференцирование всех рядов
df_diff = df_lagged.copy()

for col in df_diff.columns:
    df_diff[col] = df_diff[col].diff()

df_diff = df_diff.dropna()

# проверка стационарности после дифференцирования
print("\nПроверка после дифференцирования")
for col in df_diff.columns:
    check_stationarity(df_diff[col], col)

# деление на train и test
train = df_diff.iloc[:-test_size]
test = df_diff.iloc[-test_size:]

# выделение целевой переменной и факторов
endog_train = train["ipc"]
exog_train = train[exog_cols]

endog_test = test["ipc"]
exog_test = test[exog_cols]

# перебор параметров и диагностика
from itertools import product
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

results = []

def diagnose_model(train_endog, train_exog, order, seasonal_order):
    model = SARIMAX(
        train_endog,
        exog=train_exog,
        order=order,
        seasonal_order=seasonal_order
    ).fit(maxiter=1000, disp=False)

    residuals = model.resid

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    plot_acf(residuals, ax=axes[0], lags=24, alpha=0.05, zero=False)
    axes[0].set_title(f'ACF остатков: SARIMAX{order}{seasonal_order}')
    axes[0].set_xlabel("Лаг")
    axes[0].set_ylabel("Автокорреляция")

    plot_pacf(residuals, ax=axes[1], lags=24, alpha=0.05, method='ywm', zero=False)
    axes[1].set_title(f'PACF остатков: SARIMAX{order}{seasonal_order}')
    axes[1].set_xlabel("Лаг")
    axes[1].set_ylabel("Частичная автокорреляция")

    plt.tight_layout()
    plt.show()

    lb_test = acorr_ljungbox(residuals, lags=[1,2,12,24], return_df=True)

    print("\nТест Льюнга-Бокса:")
    print(lb_test)

    print("\nИнтерпретация:")
    for lag in [5, 10, 15, 20]:
        p_value = lb_test.loc[lag, 'lb_pvalue']

        if p_value > 0.05:
            print(f"Лаг {lag}: p-value = {p_value:.4f} > 0.05 -> автокорреляции нет")
        else:
            print(f"Лаг {lag}: p-value = {p_value:.4f} < 0.05 -> автокорреляция есть")


# параметры sarimax
order = (1, 0, 1)
seasonal_order = (2, 0, 1, 12)

# обучение модели на train
model = SARIMAX(
    endog_train,
    exog=exog_train,
    order=order,
    seasonal_order=seasonal_order
)

results = model.fit(maxiter=1000, disp=False) # чтобы метод макс правдоподобия сошелся

# прогноз на тестовую выборку
pred_diff = results.forecast(
    steps=test_size,
    exog=exog_test
)

# восстановление уровней из разностей
last_train_level = df_lagged["ipc"].iloc[-test_size - 1]
pred_level = last_train_level + pred_diff.cumsum()

# фактические значения для теста
actual_level = df["ipc"].iloc[-test_size:]

# расчет метрик качества
mae = mean_absolute_error(actual_level, pred_level)
rmse = np.sqrt(mean_squared_error(actual_level, pred_level))
mape = mean_absolute_percentage_error(actual_level, pred_level) * 100

print("\nКачество на тестовой выборке (12 последних наблюдений)")
print(f"MAE  = {mae:.3f}")
print(f"RMSE = {rmse:.3f}")
print(f"MAPE = {mape:.2f}%")

# обучение модели на полном ряду
endog_full = df_diff["ipc"]
exog_full = df_diff[exog_cols]

best_model_full = SARIMAX(
    endog_full,
    exog=exog_full,
    order=order,
    seasonal_order=seasonal_order
).fit(maxiter=1000, disp=False)

# подготовка экзогенных данных для прогноза
future_exog = df_lagged[exog_cols].iloc[-forecast_steps:].copy()

future_exog.index = pd.date_range(
    start=df_lagged.index[-1] + pd.offsets.MonthBegin(1),
    periods=forecast_steps,
    freq='MS'
)

# прогноз в разностях
forecast_diff = best_model_full.forecast(
    steps=forecast_steps,
    exog=future_exog
)

# возврат к исходному масштабу
last_actual = df["ipc"].iloc[-1]
forecast = last_actual + forecast_diff.cumsum()

# вывод прогноза
print("\nПрогноз на 6 месяцев вперед")
for d, val in zip(forecast.index, forecast):
    print(f"{d.strftime('%Y-%m')} : {val:.2f}")

print("\nПараметры модели SARIMAX")
print(f"{'Параметр':<20} {'Оценка':<15} {'p-value':<15}")
print("-" * 50)

for param in best_model_full.params.index:
    estimate = best_model_full.params[param]
    p_value = best_model_full.pvalues[param]

    print(f"{param:<20} {estimate:<15.6f} {p_value:<15.6f}")

# график исторического ряда и прогноза

plt.figure(figsize=(12, 6))

plt.plot(df.index, df["ipc"], label="Исторические данные")
plt.plot(forecast.index, forecast, label="Прогноз на 6 месяцев", marker="o", linestyle="--")

plt.axvline(x=df.index[-1], linestyle="--", alpha=0.7)

plt.title("SARIMAX: прогноз IPC на 6 месяцев вперед")
plt.xlabel("Дата")
plt.ylabel("IPC")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# графики остатков модели

residuals = best_model_full.resid

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

plot_acf(residuals, lags=24, ax=axes[0], zero=False)
axes[0].set_title("ACF остатков")
axes[0].set_xlabel("Лаг")
axes[0].set_ylabel("Автокорреляция")

plot_pacf(residuals, lags=24, ax=axes[1], method='ywm', zero=False)
axes[1].set_title("PACF остатков")
axes[1].set_xlabel("Лаг")
axes[1].set_ylabel("Частичная автокорреляция")

plt.tight_layout()
plt.show()

# вывод коэффициентов модели

print("\nКоэффициенты модели SARIMAX")
print(f"{'Параметр':<20} {'Коэффициент':<12} {'Std.Err':<12} {'z':<12} {'p-value':<12} {'Значим'}")
print("-" * 85)

for param in best_model_full.params.index:
    coef = best_model_full.params[param]
    std_err = best_model_full.bse[param]
    z_stat = coef / std_err
    p_value = best_model_full.pvalues[param]

    significant = "Да" if p_value < 0.05 else "Нет"

    print(f"{param:<20} {coef:<12.6f} {std_err:<12.6f} {z_stat:<12.4f} {p_value:<12.4f} {significant}")