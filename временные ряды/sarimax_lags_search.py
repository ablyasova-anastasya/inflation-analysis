import pandas as pd
import numpy as np
import warnings
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


results = []

warnings.filterwarnings("ignore")

for p, q, P, Q in product(range(3), range(3), range(3), range(3)):
    try:
        order = (p, 0, q)
        seasonal_order = (P, 0, Q, 12)

        with warnings.catch_warnings(record=True) as w:
            model = SARIMAX(
                endog_train,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order
            ).fit(maxiter=1000, disp=False)

            convergence_warnings = [
                warning for warning in w
                if issubclass(warning.category, ConvergenceWarning)
            ]

            if convergence_warnings:
                continue

        pred_diff = model.forecast(
            steps=test_size,
            exog=exog_test
        )

        # возврат из разностей в уровни
        last_train_level = df_lagged["ipc"].iloc[-test_size - 1]
        pred = last_train_level + pred_diff.cumsum()

        actual = df["ipc"].iloc[-test_size:]

        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = mean_absolute_percentage_error(actual, pred) * 100

        results.append({
            'p': p,
            'q': q,
            'P': P,
            'Q': Q,
            'order': order,
            'seasonal': seasonal_order,
            'aic': model.aic,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        })

        print(
            f"SARIMAX({p},0,{q})({P},0,{Q},12) | "
            f"AIC={model.aic:.2f} | "
            f"MAE={mae:.3f} | "
            f"RMSE={rmse:.3f} | "
            f"MAPE={mape:.2f}%"
        )

    except:
        continue


print("\n" + "=" * 80)
print("ТОП-20 МОДЕЛЕЙ ПО AIC:")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('aic')

for i, row in results_df.head(20).iterrows():
    print(
        f"{i + 1}. SARIMAX({row['p']},0,{row['q']})({row['P']},0,{row['Q']},12) | "
        f"AIC={row['aic']:.2f} | "
        f"MAPE={row['mape']:.2f}%"
    )

