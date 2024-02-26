import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from _trade import Trade
from _long_short_trade import TradeLongShort
import pandas as pd
from sklearn.linear_model import Ridge


class StrategyLinearRegressionMultiAgent:
    def __init__(self, asset, dif_time, data, train_duration, test_duration, model, thresold, stop_loss=None) -> None:
        self.asset = asset
        self.dif_time = dif_time
        self.data = self.data_preparation_for_execution(data)
        self.train_duration = train_duration
        self.test_duration = test_duration
        self.datasets = self.create_all_datasets(
            self.data, self.train_duration, self.test_duration)
        self.model = model
        self.thresold = thresold
        self.stop_loss = stop_loss
        pass

    def create_all_datasets(self, df, train_duration, test_duration):
        """
        Creates the train-test datasets within the given timeframe, based on specified train and test durations.

        Parameters:
            df (DataFrame): The data to be split into training and testing sets.
            train_duration (int): The number of month for each training set.
            test_duration (int): The number of months for each testing set.

        Returns:
            list: A list of tuples, where each tuple contains a training dataset and a testing dataset.
        """
        limit_date = df.index[0]
        list_of_datasets = []
        while limit_date < pd.to_datetime("2023-07-01"):
            train, test = self.train_test_split_time_series(
                df, train_duration, test_duration)

            list_of_datasets.append([train, test])
            last_date_to_train_next = test.index[-1]
            firs_date_of_next_dataset = last_date_to_train_next - \
                pd.DateOffset(months=train_duration)
            df = df[df.index >= firs_date_of_next_dataset]
            limit_date = last_date_to_train_next
        return list_of_datasets

    def train_test_split_time_series(self, df, train_duration, test_duration):

        end_train_date = df.index[0] + pd.DateOffset(months=train_duration)
        print(end_train_date)

        start_test_date = end_train_date

        end_test_date = start_test_date + pd.DateOffset(months=test_duration)

        train_set = df[df.index < end_train_date]
        test_set = df[(df.index >= start_test_date)
                      & (df.index < end_test_date)]

        return train_set, test_set

    def data_preparation_for_execution(self, data):
        """
        Prepares the data for execution by differencing and shifting to create features and targets.

        Parameters:
            data (DataFrame): The original data.

        Returns:
            DataFrame: The preprocessed data with features and target variable.
        """

        column_target = self.asset
        new_data = data.diff(self.dif_time)
        new_data["y"] = new_data[column_target].shift(-self.dif_time)
        new_data = new_data.dropna(0)
        return new_data

    def execution_of_strategy(self):
        """
        Executes the trading strategy by training the model on each dataset, predicting future values, and creating trades based on the predictions.

        Returns:
            list: A list of Trade objects representing the trades made by the strategy.
        """
        list_of_trades = []
        for i, (train_set, test_set) in enumerate(self.datasets):
            y_train = train_set["y"]
            train_set.drop(["y"], axis=1, inplace=True)

            test_set.drop(["y"], axis=1, inplace=True)
            strategy = StrategyLinearRegression(
                self.asset, self.dif_time, train_set, y_train, test_set, self.model, self.data.index, self.stop_loss)
            strategy.train()
            list_of_trades += strategy.trade_creation(-self.thresold,
                                                      self.thresold)
        return list_of_trades


class StrategyLinearRegression:

    def __init__(self, asset, dif_time, data_train, y_train, data_trade, model, index_all, stop_loss=None) -> None:
        # La data est deja clean (avec les diff, les shift, les dropna)

        self.asset = asset
        self.dif_time = dif_time
        self.data_train = data_train
        self.y_train = y_train
        self.data_trade = data_trade

        '''self.data_train = data_train.diff(self.dif_time)
        self.data_trade = data_trade.diff(self.dif_time)
        self.data_train = self.data_train.dropna(axis=0)
        self.data_trade = self.data_trade.dropna(axis=0)

        self.y_train = self.data_train[self.asset].shift(-self.dif_time)
        self.X_train = self.data_train.drop(
            columns=[self.asset]).iloc[:-self.dif_time]
        self.y_train = self.y_train.dropna()

        # self.y_trade = self.data_trade[self.asset].shift(-self.dif_time)
        self.X_trade = self.data_trade.drop(
            columns=[self.asset]).iloc[:-self.dif_time]'''

        self.model = model
        self.index_all = index_all
        self.stop_loss = stop_loss

        return

    def train(self):
        """
        Trains the regression model on the provided training data and targets.
        """

        if self.data_train.shape[0] == self.y_train.shape[0]:
            self.model.fit(self.data_train, self.y_train)

            '''predictions = self.model.predict(X)
            mse = mean_squared_error(self.y_train, predictions)  
            rmse = np.sqrt(mse)  '''
        return

    def trade_creation(self, trhesold_long, trhesold_short):
        """
        Creates trades based on the model's predictions, with separate thresholds for long and short positions.

        Parameters:
            threshold_long (float): The threshold for initiating long trades.
            threshold_short (float): The threshold for initiating short trades.

        Returns:
            list: A list of Trade objects representing the initiated trades.
        """

        predictions = self.model.predict(self.data_trade)
        predictions = pd.DataFrame(
            predictions, index=self.data_trade.index, columns=['Predicted Value'])

        list_of_trades = []
        for index, row in predictions.iterrows():

            '''prediction = row['Predicted Value']
            position = predictions.index.get_loc(index) + 1'''

            prediction = row['Predicted Value']
            if prediction > trhesold_short:
                quantity = 1
                start_date = index
                position = self.index_all.get_loc(start_date) + 1
                stop_date = self.index_all[position +
                                           self.dif_time]
                trade = Trade(self.asset, quantity, -
                              1, start_date, stop_date, self.stop_loss)
                list_of_trades.append(trade)

            if prediction < trhesold_long:

                quantity = 1
                start_date = index
                position = self.index_all.get_loc(start_date) + 1
                stop_date = self.index_all[position +
                                           self.dif_time]
                trade = Trade(self.asset, quantity, 1,
                              start_date, stop_date, self.stop_loss)
                list_of_trades.append(trade)

        return list_of_trades
