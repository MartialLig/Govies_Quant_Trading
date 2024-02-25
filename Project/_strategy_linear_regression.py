import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from _trade import Trade
from _long_short_trade import TradeLongShort
import pandas as pd
from sklearn.linear_model import Ridge


class StrategyLinearRegression:

    def __init__(self, asset, dif_time, data_train, data_trade) -> None:
        self.asset = asset
        self.dif_time = dif_time
        self.data_train = data_train.diff(self.dif_time)
        self.data_trade = data_trade.diff(self.dif_time)
        self.data_train = self.data_train.dropna(axis=0)
        self.data_trade = self.data_trade.dropna(axis=0)

        self.y_train = self.data_train[self.asset].shift(-self.dif_time)
        self.X_train = self.data_train.drop(
            columns=[self.asset]).iloc[:-self.dif_time]
        self.y_train = self.y_train.dropna()

        # self.y_trade = self.data_trade[self.asset].shift(-self.dif_time)
        self.X_trade = self.data_trade.drop(
            columns=[self.asset]).iloc[:-self.dif_time]

        self.model = Ridge()

        return

    def train(self):

        if self.X_train.shape[0] == self.y_train.shape[0]:
            self.model.fit(self.X_train, self.y_train)

            '''predictions = self.model.predict(X)
            mse = mean_squared_error(self.y_train, predictions)  
            rmse = np.sqrt(mse)  '''
        return

    def trade_creation(self, trhesold_long, trhesold_short):

        predictions = self.model.predict(self.X_trade)
        predictions = pd.DataFrame(
            predictions, index=self.X_trade.index, columns=['Predicted Value'])

        list_of_trades = []
        for index, row in predictions.iterrows():

            prediction = row['Predicted Value']
            position = predictions.index.get_loc(index) + 1
            if position + self.dif_time + 5 <= len(predictions):

                prediction = row['Predicted Value']
                if prediction > trhesold_short:
                    quantity = 1
                    start_date = index
                    position = predictions.index.get_loc(start_date) + 1
                    stop_date = predictions.index[position +
                                                  self.dif_time]
                    trade = Trade(self.asset, quantity, -
                                  1, start_date, stop_date)
                    list_of_trades.append(trade)

                if prediction < trhesold_long:

                    quantity = 1
                    start_date = index
                    position = predictions.index.get_loc(start_date) + 1
                    stop_date = predictions.index[position +
                                                  self.dif_time]
                    trade = Trade(self.asset, quantity, 1,
                                  start_date, stop_date)
                    list_of_trades.append(trade)

        return list_of_trades
