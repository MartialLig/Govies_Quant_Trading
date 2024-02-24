import pandas as pd
from _data_manager import DataManager
from _trade import Trade
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from _long_short_trade import TradeLongShort


file_path = "EGB_historical_yield.csv"
data_manager = DataManager(file_path)

trade = Trade("Germany_30y", -1, 1, "2016-01-15", "2016-01-21")

trade.daily_Pand_L(data_manager.data)

print("hello")
print(trade.p_and_l_data)
print(trade.p_and_l)

trade_long_short = TradeLongShort(
    "Germany_30y", "Belgium_30y", 1, "2016-01-15", "2016-01-21")
trade_long_short.daily_Pand_L(data_manager.data)

print("hello")
print(trade_long_short.p_and_l_data_detailed)
print(trade_long_short.p_and_l_data)
print(trade_long_short.p_and_l)
