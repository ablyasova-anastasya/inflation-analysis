import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates

df = pd.read_excel('/Users/nastya/Documents/диплом/временные ряды/ipc.xlsx')
df['date'] = pd.to_datetime(df['date'])
dates = [x.strftime('%Y-%m') for x in df['date']]
ind = [i for i in range(len(dates))]
print(f"\n--- Проверка стационарности ---")

# Тест Дики-Фуллера
result = adfuller(df['ipc'])
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')

if result[1] <= 0.05:
    print("Ряд стационарен (отвергаем H0)")
else:
    print("Ряд не стационарен (не отвергаем H0)")

from statsmodels.tsa.stattools import kpss


def kpss_test(timeseries):
    print('Результаты KPSS теста:')
    kpss_stat, p_value, lags, critical_values = kpss(timeseries, regression='c')

    print(f'KPSS Statistic: {kpss_stat:.4f}')
    print(f'p-value: {p_value:.4f}')
    print(f'Lags: {lags}')

    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value:.3f}')

    if p_value < 0.05:
        print("Результат: Ряд не стационарен (отвергаем H0)")
    else:
        print("Результат: Ряд стационарен (не отвергаем H0)")

    return kpss_stat, p_value


print('-' * 100)

kpss_stat, kpss_pvalue = kpss_test(df['ipc'])
print('-' * 100)

fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(df['date'], df['ipc'])
ax.set_title('Временной ряд ИПЦ')
ax.set_xlabel("Дата")
ax.set_ylabel("ИПЦ")

ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.show()

# df1 = []
# ipc = df['ipc'].tolist()
# for i in range(1, len(df['ipc'])):
#     df1.append(ipc[i] - ipc[i - 1])
ipc_diff=df['ipc'].diff(periods=1).dropna()
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(df['date'][1:], ipc_diff)
ax.set_title('Временной ряд после дифференцирования на 1')
ax.set_xlabel("Дата")
ax.set_ylabel("Изменение ИПЦ")

ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.show()


fig, ax = plt.subplots(figsize=(18, 8))
plot_acf(ipc_diff,
         lags=30,  # сколько лагов показывать (можно менять)
         alpha=0.05,  # уровень значимости для доверительного интервала
         title='Автокорреляционная функция (ACF) после дифференцирования на 1',
         zero=False,
         ax=ax)  # не показывать лаг 0 (всегда равен 1)
ax.set_xlabel("Лаг")
ax.set_ylabel("Автокорреляция")
plt.show()
fig, ax = plt.subplots(figsize=(18, 8))
plot_pacf(ipc_diff,
          lags=30,
          alpha=0.05,
          title='Частная автокорреляционная функция (PACF) после дифференцирования на 1',
          zero=False,
          ax=ax
          )
ax.set_xlabel("Лаг")
ax.set_ylabel("Частная автокорреляция")
plt.show()


# Тест Дики-Фуллера
result = adfuller(ipc_diff)
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')

if result[1] <= 0.05:
    print("Ряд стационарен (отвергаем H0)")
else:
    print("Ряд не стационарен (не отвергаем H0)")


print('-' * 100)

kpss_stat, kpss_pvalue = kpss_test(ipc_diff)
print('-' * 100)


ipc_diff_12=ipc_diff.diff(periods=12).dropna()
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot(df['date'][13:], ipc_diff_12)
ax.set_title('Временной ряд после дифференцирования на 12')
ax.grid(True)

ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.show()


# Тест Дики-Фуллера
result = adfuller(ipc_diff_12)
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')

if result[1] <= 0.05:
    print("Ряд стационарен (отвергаем H0)")
else:
    print("Ряд не стационарен (не отвергаем H0)")


print('-' * 100)

kpss_stat, kpss_pvalue = kpss_test(ipc_diff_12)
print('-' * 100)



fig, ax = plt.subplots(figsize=(18, 8))
plot_acf(ipc_diff_12,
         lags=30,  # сколько лагов показывать (можно менять)
         alpha=0.05,  # уровень значимости для доверительного интервала
         title='Автокорреляционная функция (ACF) после дифференцирования на 12',
         zero=False,
         ax=ax)  # не показывать лаг 0 (всегда равен 1)
plt.show()
fig, ax = plt.subplots(figsize=(18, 8))
plot_pacf(ipc_diff_12,
          lags=30,
          alpha=0.05,
          title='Частная автокорреляционная функция (PACF) после дифференцирования на 12',
          zero=False,
          ax=ax
          )
plt.show()
