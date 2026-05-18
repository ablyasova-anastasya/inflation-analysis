import pandas as pd
import numpy as np

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def load_series(path, value_name):
    df = pd.read_excel(path, parse_dates=['date'])
    df = df[['date', df.columns[1]]]
    df.columns = ['date', value_name]

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').asfreq('MS')
    return df


base_path = "/Users/nastya/Documents/диплом/временные ряды/"

ipc = load_series(base_path + "ipc.xlsx", "ipc")
ipp = load_series(base_path + "ipp.xlsx", "ipp")
dollar = load_series(base_path + "dollar.xlsx", "dollar")
unemployment = load_series(base_path + "unemployment.xlsx", "unemployment")
salary = load_series(base_path + "salary.xlsx", "salary")
trading = load_series(base_path + "trading.xlsx", "trading")

df = pd.concat([ipc, ipp, trading, unemployment, salary, dollar], axis=1)
df = df.dropna()

# проверка стационарности до дифференцирования
print("Проверка стационарности исходных рядов:")

for col in df.columns:
    # Тест Дики-Фуллера
    adf_res = adfuller(df[col].dropna())
    print(f"\n{col.upper()}:")
    print(f"  ADF: statistic={adf_res[0]:.4f}, p-value={adf_res[1]:.4f}")

    # Тест KPSS
    kpss_res = kpss(df[col].dropna(), regression='c', nlags='auto')
    print(f"  KPSS: statistic={kpss_res[0]:.4f}, p-value={kpss_res[1]:.4f}")

    adf_stationary = adf_res[1] < 0.05
    kpss_stationary = kpss_res[1] > 0.05

    if adf_stationary and kpss_stationary:
        print("ряд стационарен")
    elif not adf_stationary and not kpss_stationary:
        print("ряд нестационарен")
    else:
        print("результаты тестов отличаются")


df1 = df.copy()

diff_data = df1.diff().dropna()

# проверка стационарности после дифференцирования
print("\n" + "-" * 60)
print("Проверка стационарности после дифференцирования: ")


for col in diff_data.columns:
    # Тест Дики-Фуллера
    adf_res = adfuller(diff_data[col].dropna())
    print(f"\n{col.upper()}:")
    print(f"  ADF: statistic={adf_res[0]:.4f}, p-value={adf_res[1]:.4f}")

    # Тест KPSS
    kpss_res = kpss(diff_data[col].dropna(), regression='c', nlags='auto')
    print(f"  KPSS: statistic={kpss_res[0]:.4f}, p-value={kpss_res[1]:.4f}")

    adf_stationary = adf_res[1] < 0.05
    kpss_stationary = kpss_res[1] > 0.05
    if adf_stationary and kpss_stationary:
        print("ряд стационарен")
    elif not adf_stationary and not kpss_stationary:
        print( "ряд нестационарен")
    else:
        print( "результаты тестов отличаются")

# train / test split
test_size = 12

train = diff_data.iloc[:-test_size]
test = diff_data.iloc[-test_size:]

# обучение на train
model = VAR(train)

lag_order = model.select_order(18)
p = lag_order.selected_orders['aic']
print(lag_order.summary())

var_model = model.fit(p)

# прогноз на test
last_values = train.values[-p:]
forecast_diff_test = var_model.forecast(last_values, steps=test_size)

forecast_diff_test_df = pd.DataFrame(
    forecast_diff_test,
    columns=diff_data.columns,
    index=test.index
)

# интегрирование прогноза test
last_real = df1.iloc[:-test_size].iloc[-1]

forecast_levels = []
current = last_real.copy()

for i in range(test_size):
    current = current + forecast_diff_test_df.iloc[i]
    forecast_levels.append(current.copy())

forecast_levels_df = pd.DataFrame(forecast_levels, index=test.index)

# метрики качества
y_true = df1.iloc[-test_size:]
errors = []

for col in df1.columns:
    y_pred = forecast_levels_df[col]

    mae = mean_absolute_error(y_true[col], y_pred)
    rmse = np.sqrt(mean_squared_error(y_true[col], y_pred))
    mape = mean_absolute_percentage_error(y_true[col], y_pred) * 100

    errors.append([col, mae, rmse, mape])

errors_df = pd.DataFrame(errors, columns=["variable", "MAE", "RMSE", "MAPE"])

print("\n Ошибки по test (12 месяцев):")
print(errors_df)

# переобучение на всех данных
model_full = VAR(diff_data)
var_model = model_full.fit(p)

# прогноз на 6 месяцев вперед
forecast_steps = 6

last_values = diff_data.values[-p:]
forecast_diff = var_model.forecast(last_values, steps=forecast_steps)

forecast_diff_df = pd.DataFrame(forecast_diff, columns=diff_data.columns)

# интегрирование прогноза
last_real = df1.iloc[-1]

forecast_levels = []
current = last_real.copy()

for i in range(forecast_steps):
    current = current + forecast_diff_df.iloc[i]
    forecast_levels.append(current.copy())

forecast_levels_df = pd.DataFrame(forecast_levels)
forecast_levels_df.index = pd.date_range(
    start=df1.index[-1] + pd.offsets.MonthBegin(),
    periods=forecast_steps,
    freq='MS'
)


# результаты
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

ipc_table = pd.concat([
    var_model.params["ipc"],
    var_model.bse["ipc"],
    var_model.tvalues["ipc"],
    var_model.pvalues["ipc"]
], axis=1)

ipc_table.columns = ["coef", "std_err", "t_stat", "p_value"]

print("\n Результаты: ")
print(ipc_table)

# прогноз на 6 месяцев
print("\n Прогноз ИПЦ")
print(forecast_levels_df["ipc"])

# тест Льюнга-Бокса
resid = var_model.resid["ipc"]

lb_test = acorr_ljungbox(resid, lags=[12, 24], return_df=True)

print("\n тест Льюнга-Бокса")
print(lb_test)

# стабильность модели
print("\nVAR stable:", var_model.is_stable())

for col in forecast_levels_df.columns:
    print(f"\n{col.upper()}:")
    print(forecast_levels_df[col].to_string())

resid_ipc = var_model.resid['ipc']

fig, ax = plt.subplots(1, 2, figsize=(14,4))

plot_acf(resid_ipc, lags=40, ax=ax[0], alpha=0.05, zero=False)
ax[0].set_title("ACF остатков ИПЦ")
ax[0].set_xlabel("Лаг")
ax[0].set_ylabel("Автокорреляция")

plot_pacf(resid_ipc, lags=40, ax=ax[1], alpha=0.05, zero=False)
ax[1].set_title("PACF остатков ИПЦ")
ax[1].set_xlabel("Лаг")
ax[1].set_ylabel("Частная автокорреляция")

plt.tight_layout()
plt.show()