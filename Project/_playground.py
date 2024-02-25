import pandas as pd
from _data_manager import DataManager
from _trade import Trade
from _long_short_trade import TradeLongShort
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from _backtesting import Backtest
from _strategy_linear_regression import StrategyLinearRegression

file_path = "EGB_historical_yield.csv"
data_manager = DataManager(file_path)

trade = Trade("Germany_30y", -1, 1, "2016-01-15", "2016-01-21")

trade.daily_Pand_L(data_manager.data)

print("hello")
print(trade.p_and_l_data)
print(trade.p_and_l)

trade_long_short = TradeLongShort(
    "Germany_30y", "Belgium_30y", 1, "2016-02-23", "2016-08-29")
trade_long_short.daily_Pand_L(data_manager.data)

print("hello")
print(trade_long_short.p_and_l_data_detailed)
print(trade_long_short.p_and_l_data)
print(trade_long_short.p_and_l)

list_of_trade = [trade, trade_long_short]

back = Backtest(list_of_trade, data_manager.data)
back.gather_all_trades()
back.p_and_l_detailed_dataset.to_clipboard()


new_data = data_manager.data[data_manager.data.index < "2017-01-01"]

new_data_trades = data_manager.data[
    (data_manager.data.index > "2017-01-01") & (data_manager.data.index < "2018-01-01")
]

strategy = StrategyLinearRegression(
    "Austria_5y", 5, new_data, new_data_trades)
print("hello")
strategy.train()


list_of_trade = strategy.trade_creation(-2, 2)


back = Backtest(list_of_trade, new_data_trades)
back.gather_all_trades()
back.plot_p_and_l()
print("hend")
