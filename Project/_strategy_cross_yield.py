import pandas as pd
from _long_short_trade import TradeLongShort


class StrategyCrossYield:
    def __init__(self, data_spread_yield, time_to_keep_the_asset_if_not_cross=5, stop_loss=None) -> None:
        self.data_spread_yield = data_spread_yield
        self.signal = self.detect_sign_changes(self.data_spread_yield)
        self.time_to_keep_the_asset_if_not_cross = time_to_keep_the_asset_if_not_cross
        self.stop_loss = stop_loss
        pass

    def detect_sign_changes(self, df):
        changes = pd.DataFrame(index=df.index, columns=df.columns)

        for col in df.columns:
            changes[col] = 0

            for i in range(1, len(df)):
                if df[col].iloc[i-1] >= 0 and df[col].iloc[i] < 0:
                    changes[col].iloc[i] = -1
                elif df[col].iloc[i-1] < 0 and df[col].iloc[i] >= 0:
                    changes[col].iloc[i] = 1

        return changes

    def execution_of_strategy(self):
        trades_creators = TradeCreationLongShort(
            self.signal, self.time_to_keep_the_asset_if_not_cross, self.stop_loss)
        list_of_trades = trades_creators.creation_list_of_trades()
        list_of_trades = sorted(list_of_trades, key=lambda x: x.start_date)

        return list_of_trades


class TradeCreationLongShort():
    def __init__(self, list_of_orders, time_to_keep_the_asset_if_not_cross=5, stop_loss=None) -> None:
        self.list_of_orders = list_of_orders
        self.list_of_trades = []
        self.time_to_keep_the_asset_if_not_cross = time_to_keep_the_asset_if_not_cross
        self.stop_loss = stop_loss
        pass

    def creation_list_of_trades(self):
        for col in self.list_of_orders.columns:
            skip_until_index = None  # Flag pour ignorer les vérifications jusqu'à un certain index
            for index, row in self.list_of_orders.iterrows():
                if skip_until_index is not None and index <= skip_until_index:
                    continue

                if row[col] != 0:
                    quantity = 1
                    split_world = col.split("-")
                    if row[col] == 1:
                        long_asset = split_world[0]
                        short_asset = split_world[1]

                    elif row[col] == -1:
                        long_asset = split_world[1]
                        short_asset = split_world[0]
                    start_date = index

                    stop_date = None
                    future_indexes = self.list_of_orders.index.get_loc(
                        index) + 1
                    for i in range(1, self.time_to_keep_the_asset_if_not_cross):
                        if future_indexes + i < len(self.list_of_orders):
                            next_index = self.list_of_orders.index[future_indexes + i]
                            if self.list_of_orders.at[next_index, col] != 0:
                                stop_date = next_index
                                skip_until_index = next_index
                                break
                    if stop_date is None:
                        if future_indexes + 5 < len(self.list_of_orders):
                            stop_date = self.list_of_orders.index[future_indexes + 5]
                    trade = TradeLongShort(
                        long_asset, short_asset, quantity, start_date, stop_date, self.stop_loss)
                    self.list_of_trades.append(trade)
        return self.list_of_trades
