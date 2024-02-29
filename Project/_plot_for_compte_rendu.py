# This file is simply the notebook in Python format. The reason for this is because I encounter bugs
# with the display of graphs for the strategies in Jupyter Notebook—the graphs disappear over time.
# Exporting it in this format allows all graphs to be opened in a web browser, avoiding this issue.


import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from _data_manager import DataManager
from _trade import Trade
from _long_short_trade import TradeLongShort
from _backtesting import Backtest
from _strategy_linear_regression import StrategyLinearRegression, StrategyLinearRegressionMultiAgent
from _strategy_cross_yield import StrategyCrossYield
from _trade_filter import TradeFilter
from _strategy_comparator import StratgiesComparetor
from _strategy_linear_optimisation_time import AssetSelectioByModel, OptimisationTimeRegressison


pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore')


file_path = 'EGB_historical_yield.csv'
data_manager = DataManager(file_path)


data_manager.data.shape

data_manager.data.head()

data_manager.data.dtypes

data_manager.data.isnull().sum().sort_values().head(3)

data_manager.data.describe()

data_manager.data.std(axis=0).sort_values()

corr = data_manager.data.corr()


data_spread_yield = data_manager.spread_yield
time_to_keep_the_asset_if_not_cross = [5, 10, 20, 40, 60, 80]
stop_loss = None

list_of_strategy = []
list_of_strategy_name = []
for time_to_keep_if_not_cross in time_to_keep_the_asset_if_not_cross:
    cross_strategy = StrategyCrossYield(
        data_spread_yield, time_to_keep_if_not_cross, stop_loss)
    liste_trades = cross_strategy.execution_of_strategy()

    title = f"Profit and Loss Over Time - {time_to_keep_if_not_cross} days - Cross Strategy"
    backtest_cross_strategy = Backtest(liste_trades, data_manager.data)
    backtest_cross_strategy.gather_all_trades()
    backtest_cross_strategy.plot_p_and_l(title)
    list_of_strategy.append(backtest_cross_strategy)
    list_of_strategy_name.append(f"{time_to_keep_if_not_cross} days")


comparator = StratgiesComparetor(list_of_strategy, list_of_strategy_name)
comparator.compare_results_of_strategies()

data_spread_yield = data_manager.spread_yield
time_to_keep_the_asset_if_not_cross = [60, 80]
stop_loss = [-20, -50, -100]

list_of_strategy = []
list_of_strategy_name = []
for stop in stop_loss:
    for time_to_keep_if_not_cross in time_to_keep_the_asset_if_not_cross:

        cross_strategy = StrategyCrossYield(
            data_spread_yield, time_to_keep_if_not_cross, stop)
        liste_trades = cross_strategy.execution_of_strategy()

        title = f"Profit and Loss Over Time - {time_to_keep_if_not_cross} days, {stop} stop loss - Cross Strategy"
        backtest_cross_strategy = Backtest(liste_trades, data_manager.data)
        backtest_cross_strategy.gather_all_trades()
        backtest_cross_strategy.plot_p_and_l(title)
        list_of_strategy.append(backtest_cross_strategy)
        list_of_strategy_name.append(
            f"{time_to_keep_if_not_cross} days, {stop} stop loss ")

comparator = StratgiesComparetor(list_of_strategy, list_of_strategy_name)
comparator.compare_results_of_strategies()

corr = data_manager.data.diff(5).corr()

plt.figure(figsize=(25, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', cbar=True,
            square=True, fmt='.2f', annot_kws={'size': 10})
plt.title('Matrice de Corrélation')
plt.show()

data_manager.data.diff(5).corr().sum().sort_values(ascending=False).head(5)

start_date = '2016-01-01'
end_date = '2017-03-01'
model = LinearRegression()

asset_selector = AssetSelectioByModel(
    data_manager.data, start_date, end_date, model)
asset_selector.train_and_evaluate().head(10)

test = OptimisationTimeRegressison()
premiere_valeurs = [1, 2, 3, 6, 9, 12, 18]
seconde_valeurs = [1, 2, 3, 4, 5, 6, 9, 12]
liste_temps = [(x, y) for x in premiere_valeurs for y in seconde_valeurs]
number_to_diff = 5
column_target = "Austria_10y"
new_data = data_manager.data.diff(number_to_diff)
new_data[column_target] = new_data[column_target].shift(-number_to_diff)
new_data = new_data.dropna(0)

data1, data2 = test.find_optimal_value(new_data, liste_temps, column_target)
data1.sort_values("Test_R2", ascending=False)

asset = "Austria_5y"
number_of_day_dif = 5
perdiod_of_training = 6
period_of_trade = 1
model = LinearRegression()
triger = 0.001


firs_attempt = StrategyLinearRegressionMultiAgent(
    asset, number_of_day_dif, data_manager.data, perdiod_of_training, period_of_trade, model, triger)
liste_trades = firs_attempt.execution_of_strategy()

backtest_first_attempt = Backtest(liste_trades, data_manager.data)
backtest_first_attempt.gather_all_trades()
backtest_first_attempt.data_with_adjusted_index()
backtest_first_attempt.plot_p_and_l("P&L - First Attempt - Strategy 2")
backtest_first_attempt.aggregate_results()

asset = "Austria_5y"
number_of_day_dif = 5
perdiod_of_training = 6
period_of_trade = 1
model = LinearRegression()
triger = 2


firs_attempt = StrategyLinearRegressionMultiAgent(
    asset, number_of_day_dif, data_manager.data, perdiod_of_training, period_of_trade, model, triger)
liste_trades = firs_attempt.execution_of_strategy()

backtest_first_attempt = Backtest(liste_trades, data_manager.data)
backtest_first_attempt.gather_all_trades()
backtest_first_attempt.data_with_adjusted_index()
backtest_first_attempt.plot_p_and_l("P&L - Thresold feature - Strategy 2")
backtest_first_attempt.aggregate_results()

asset = "Austria_5y"
number_of_day_dif = 5
perdiod_of_training = 6
period_of_trade = 1
model = LinearRegression()
triger = 2
stop_loss = -10


firs_attempt = StrategyLinearRegressionMultiAgent(
    asset, number_of_day_dif, data_manager.data, perdiod_of_training, period_of_trade, model, triger, stop_loss)
liste_trades = firs_attempt.execution_of_strategy()

backtest_first_attempt = Backtest(liste_trades, data_manager.data)
backtest_first_attempt.gather_all_trades()
backtest_first_attempt.data_with_adjusted_index()
backtest_first_attempt.plot_p_and_l("P&L - Stop loss feature - Strategy 2")
backtest_first_attempt.aggregate_results()

asset = "Austria_5y"
number_of_day_dif = 5
perdiod_of_training = 6
period_of_trade = 1
model = LinearRegression()
triger = 2
stop_loss = -10
var_filter = 5

firs_attempt = StrategyLinearRegressionMultiAgent(
    asset, number_of_day_dif, data_manager.data, perdiod_of_training, period_of_trade, model, triger, stop_loss)
liste_trades = firs_attempt.execution_of_strategy()

trade_filter = TradeFilter(liste_trades, var_filter)
liste_trades = trade_filter.filter_by_vol(
    data_manager.data_std_30D_by_country["Austria"])

backtest_first_attempt = Backtest(liste_trades, data_manager.data)
backtest_first_attempt.gather_all_trades()
backtest_first_attempt.data_with_adjusted_index()
backtest_first_attempt.plot_p_and_l(
    "P&L - Variance filter feature - Strategy 2")
backtest_first_attempt.aggregate_results()

dict_model = {"lin_reg": LinearRegression(), "ridge": Ridge(), "lasso": Lasso(
), "rand_for": RandomForestRegressor(), "xgb": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)}
asset = "Austria_5y"
number_of_day_dif = 5
perdiod_of_training = 6
period_of_trade = 1
triger = 3
stop_loss = -10


list_of_strategy = []
list_of_strategy_name = []

for model_name, model_to_test in dict_model.items():
    strategy = StrategyLinearRegressionMultiAgent(
        asset, number_of_day_dif, data_manager.data, perdiod_of_training, period_of_trade, model_to_test, triger, stop_loss)
    liste_trades = strategy.execution_of_strategy()

    title = f"Profit and Loss - {model_name} - Strategy 2"
    backtest_different_model = Backtest(liste_trades, data_manager.data)
    backtest_different_model.gather_all_trades()
    backtest_different_model.data_with_adjusted_index()
    backtest_different_model.plot_p_and_l(title)
    list_of_strategy.append(backtest_different_model)
    list_of_strategy_name.append(f"{model_name}")


comparator = StratgiesComparetor(list_of_strategy, list_of_strategy_name)
comparator.compare_results_of_strategies()


model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
end_date = pd.to_datetime("2018-01-01")

asset = "Austria_5y"
number_of_day_dif = 5
perdiod_of_training = 6
period_of_trade = 1
stop_loss = -10
list_trigger = [0.1, 0.5, 1, 2, 3, 4, 5, 7, 10]
list_of_strategy = []
list_of_strategy_name = []

for trigger in tqdm(list_trigger):
    strategy = StrategyLinearRegressionMultiAgent(
        asset, number_of_day_dif, data_manager.data, perdiod_of_training, period_of_trade, model, trigger, stop_loss, end_date=end_date)
    liste_trades = strategy.execution_of_strategy()

    title = f"{trigger} trig"
    backtest_different_model = Backtest(liste_trades, data_manager.data)
    backtest_different_model.gather_all_trades()
    backtest_different_model.data_with_adjusted_index()
    list_of_strategy.append(backtest_different_model)
    list_of_strategy_name.append(title)


comparator = StratgiesComparetor(list_of_strategy, list_of_strategy_name)
comparator.compare_results_of_strategies()

trigger = 3
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
end_date = pd.to_datetime("2018-01-01")


number_of_day_dif_list = [5, 10, 20]
asset = "Austria_5y"
perdiod_of_training = 6
period_of_trade = 1
stop_loss = -10


list_of_strategy = []
list_of_strategy_name = []

for number_of_day_dif in tqdm(number_of_day_dif_list):
    strategy = StrategyLinearRegressionMultiAgent(
        asset, number_of_day_dif, data_manager.data, perdiod_of_training, period_of_trade, model, trigger, stop_loss, end_date=end_date)
    liste_trades = strategy.execution_of_strategy()

    title = f"{number_of_day_dif} day_dif"
    backtest_different_model = Backtest(liste_trades, data_manager.data)
    backtest_different_model.gather_all_trades()
    backtest_different_model.data_with_adjusted_index()
    list_of_strategy.append(backtest_different_model)
    list_of_strategy_name.append(title)


comparator = StratgiesComparetor(list_of_strategy, list_of_strategy_name)
comparator.compare_results_of_strategies()

number_of_day_dif = 20
trigger = 3
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
end_date = pd.to_datetime("2018-01-01")


asset = "Austria_5y"
perdiod_of_training_list = [1, 3, 6, 9, 12]
period_of_trade = 1
stop_loss = -10
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

list_of_strategy = []
list_of_strategy_name = []

for perdiod_of_training in tqdm(perdiod_of_training_list):
    strategy = StrategyLinearRegressionMultiAgent(
        asset, number_of_day_dif, data_manager.data, perdiod_of_training, period_of_trade, model, trigger, stop_loss, end_date=end_date)
    liste_trades = strategy.execution_of_strategy()

    title = f"{perdiod_of_training} size_train"
    backtest_different_model = Backtest(liste_trades, data_manager.data)
    backtest_different_model.gather_all_trades()
    backtest_different_model.data_with_adjusted_index()
    list_of_strategy.append(backtest_different_model)
    list_of_strategy_name.append(title)


comparator = StratgiesComparetor(list_of_strategy, list_of_strategy_name)
comparator.compare_results_of_strategies()

trigger = 3
number_of_day_dif = 20
perdiod_of_training = 12
period_of_trade = 1
end_date = pd.to_datetime("2018-01-01")


asset = "Austria_5y"
stop_loss_list = [-3, -5, -7, -10, -20]
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)

list_of_strategy = []
list_of_strategy_name = []

for stop_loss in tqdm(stop_loss_list):
    strategy = StrategyLinearRegressionMultiAgent(
        asset, number_of_day_dif, data_manager.data, perdiod_of_training, period_of_trade, model, trigger, stop_loss, end_date=end_date)
    liste_trades = strategy.execution_of_strategy()

    title = f"{stop_loss} stoploss"
    backtest_different_model = Backtest(liste_trades, data_manager.data)
    backtest_different_model.gather_all_trades()
    backtest_different_model.data_with_adjusted_index()
    list_of_strategy.append(backtest_different_model)
    list_of_strategy_name.append(title)


comparator = StratgiesComparetor(list_of_strategy, list_of_strategy_name)
comparator.compare_results_of_strategies()

trigger = 3
number_of_day_dif = 20
perdiod_of_training = 12
period_of_trade = 1
stop_loss = -10
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)


list_asset = ["Austria_5y", "Austria_10y", "Germany_5y",
              "Germany_30y", "France_30y", "Netherlands_30y", "Germany_10y"]

list_of_strategy = []
list_of_strategy_name = []

for asset in tqdm(list_asset):
    strategy = StrategyLinearRegressionMultiAgent(
        asset, number_of_day_dif, data_manager.data, perdiod_of_training, period_of_trade, model, trigger, stop_loss)
    liste_trades = strategy.execution_of_strategy()

    title = f"{asset}"
    backtest_different_model = Backtest(liste_trades, data_manager.data)
    backtest_different_model.gather_all_trades()
    backtest_different_model.data_with_adjusted_index()
    list_of_strategy.append(backtest_different_model)
    list_of_strategy_name.append(title)


comparator = StratgiesComparetor(list_of_strategy, list_of_strategy_name)
comparator.compare_results_of_strategies()

comparator.plot_everything()

trigger = 3
number_of_day_dif = 20
perdiod_of_training = 1
period_of_trade = 1
stop_loss = -10
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)


list_asset = ["Austria_5y", "Austria_10y", "Germany_5y",
              "Germany_30y", "France_30y", "Netherlands_30y", "Germany_10y"]

list_of_strategy = []
list_of_strategy_name = []

for asset in tqdm(list_asset):
    strategy = StrategyLinearRegressionMultiAgent(
        asset, number_of_day_dif, data_manager.data, perdiod_of_training, period_of_trade, model, trigger, stop_loss)
    liste_trades = strategy.execution_of_strategy()

    title = f"{asset}"
    backtest_different_model = Backtest(liste_trades, data_manager.data)
    backtest_different_model.gather_all_trades()
    backtest_different_model.data_with_adjusted_index()
    list_of_strategy.append(backtest_different_model)
    list_of_strategy_name.append(title)


comparator = StratgiesComparetor(list_of_strategy, list_of_strategy_name)
comparator.compare_results_of_strategies()

comparator.plot_everything()

comparator.calculate_correlations()
