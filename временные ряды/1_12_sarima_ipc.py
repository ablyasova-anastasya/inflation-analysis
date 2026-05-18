import numpy as np
import pandas as pd
import warnings
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

df = pd.read_excel('/Users/nastya/Documents/диплом/временные ряды/ipc.xlsx')
df['date'] = pd.to_datetime(df['date'])

series = df['ipc']
dates = df['date']

test_size = 12
forecast_steps = 6

train = series[:-test_size]
test = series[-test_size:]

results = []

warnings.filterwarnings("ignore")

for p, q, P, Q in product(range(3), range(3), range(3), range(3)):
    try:
        order = (p, 1, q)
        seasonal_order = (P, 0, Q, 12)

        with warnings.catch_warnings(record=True) as w:
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order
            ).fit(disp=False)

            convergence_warnings = [
                warning for warning in w
                if issubclass(warning.category, ConvergenceWarning)
            ]

            if convergence_warnings:
                continue

        pred = model.forecast(steps=test_size)

        mae = mean_absolute_error(test, pred)
        rmse = np.sqrt(mean_squared_error(test, pred))
        mape = mean_absolute_percentage_error(test, pred) * 100

        results.append({
            'p': p, 'q': q, 'P': P, 'Q': Q,
            'order': order,
            'seasonal': seasonal_order,
            'aic': model.aic,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        })

        print(
            f"SARIMA({p},1,{q})({P},0,{Q},12) | AIC={model.aic:.2f} | MAE={mae:.3f} | RMSE={rmse:.3f} | MAPE={mape:.2f}%")

    except:
        continue

print("\n" + "=" * 80)
print("ТОП-20 МОДЕЛЕЙ ПО AIC:")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('aic')

for i, row in results_df.head(20).iterrows():
    print(
        f"{i + 1}. SARIMA({row['p']},1,{row['q']})({row['P']},0,{row['Q']},12) | AIC={row['aic']:.2f} | MAPE={row['mape']:.2f}%")


def diagnose_model(train_data, order, seasonal_order):
    model = SARIMAX(
        train_data,
        order=order,
        seasonal_order=seasonal_order
    ).fit(disp=False)

    residuals = model.resid

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(residuals, ax=axes[0], lags=24, alpha=0.05,zero=False,)
    axes[0].set_title(f'ACF остатков: SARIMA{order}{seasonal_order}')
    axes[0].set_xlabel("Лаг")
    axes[0].set_ylabel("Автокорреляция")
    plot_pacf(residuals, ax=axes[1], lags=24, alpha=0.05, method='ywm',zero=False)
    axes[1].set_title(f'PACF остатков: SARIMA{order}{seasonal_order}')
    axes[1].set_xlabel("Лаг")
    axes[1].set_ylabel("Частичная автокорреляция")
    plt.tight_layout()
    plt.show()

    lb_test = acorr_ljungbox(residuals, lags=[1, 2, 12, 24], return_df=True)
    print("\nТест Льюиса-Бокса:")
    print(lb_test)

    print("\nИнтерпретация:")
    for lag in [1, 2, 12, 24]:
        p_value = lb_test.loc[lag, 'lb_pvalue']
        if p_value > 0.05:
            print(f"Лаг {lag}: p-value = {p_value:.4f} > 0.05 -> автокорреляции нет")
        else:
            print(f"Лаг {lag}: p-value = {p_value:.4f} < 0.05 -> автокорреляция есть")



diagnose_model(
    train_data=train,
    order=(0, 1, 2),  # (p, d, q)
    seasonal_order=(2, 0, 2, 12)  # (P, D, Q, period)
)

diagnose_model(
    train_data=train,
    order=(0, 1, 2),  # (p, d, q)
    seasonal_order=(0, 0, 1, 12)  # (P, D, Q, period)
)


# best_order = (0, 1, 2)
# best_seasonal = (2, 0, 2, 12)

# без незначимых
best_order = (0, 1, 2)
best_seasonal = (0, 0, 1, 12)

model = SARIMAX(
    train,
    order=best_order,
    seasonal_order=best_seasonal,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

pred = model.forecast(steps=test_size)

mae = mean_absolute_error(test, pred)
rmse = np.sqrt(mean_squared_error(test, pred))
mape = mean_absolute_percentage_error(test, pred) * 100

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТЫ МОДЕЛИ SARIMA(0,1,2)(2,0,2,12)")
print("=" * 80)

print("\nСтатистическая значимость коэффициентов:")
print("=" * 80)
print(f"{'Параметр':<15} {'Коэффициент':<12} {'Ст. ошибка':<12} {'z-статистика':<12} {'p-value':<12} {'Значим'}")
print("-" * 80)

for param, coef in model.params.items():
    std_err = model.bse[param]
    z_stat = coef / std_err
    p_val = model.pvalues[param]
    significant = "Да" if p_val < 0.05 else "Нет"

    # Правильное определение имени параметра
    if param == 'ma.L1':
        name = "θ₁"
    elif param == 'ma.L2':
        name = "θ₂"
    elif param == 'ar.S.L12':
        name = "Φ₁"
    elif param == 'ar.S.L24':
        name = "Φ₂"
    elif param == 'ma.S.L12':
        name = "Θ₁"
    elif param == 'ma.S.L24':
        name = "Θ₂"
    elif param == 'sigma2':
        name = "σ²"
    else:
        name = param

    print(f"{name:<15} {coef:<12.6f} {std_err:<12.6f} {z_stat:<12.4f} {p_val:<12.4f} {significant}")

print(f"\nКритерии качества на тестовой выборке (12 месяцев):")
print(f"  AIC  = {model.aic:.2f}")
print(f"  MAE  = {mae:.3f}")
print(f"  RMSE = {rmse:.3f}")
print(f"  MAPE = {mape:.2f}%")

best_model_full = SARIMAX(
    series,
    order=best_order,
    seasonal_order=best_seasonal
).fit(disp=False)

future_forecast = best_model_full.forecast(steps=forecast_steps)

print(f"\nПрогноз на {forecast_steps} месяцев вперед:")
print("-" * 50)

future_dates = pd.date_range(
    start=dates.iloc[-1] + pd.offsets.MonthBegin(1),
    periods=forecast_steps,
    freq='MS'
)

for d, val in zip(future_dates, future_forecast):
    print(f"  {d.strftime('%Y-%m')} : {val:.2f}")


