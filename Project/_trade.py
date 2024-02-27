

class Trade():
    def __init__(self, asset, quantity, long_short, start_date, end_date, stop_loss=None) -> None:
        self.asset = asset
        self.quantity = quantity
        self.long_short = long_short
        self.start_date = start_date
        self.end_date = end_date
        self.stop_loss = stop_loss
        self.take_profit = None  # pas coder
        self.p_and_l_data = None
        self.p_and_l = None
        self.win_trade = None
        pass

    def daily_Pand_L(self, data):
        """
        Calculates the daily profit and loss (P&L) for the trade.

        Parameters:
            data (DataFrame): The market data, indexed by date, containing asset prices.
        """
        # Cas sans stop_loss and take_profit
        start_value = data.loc[self.start_date, self.asset]
        smaller_dataset = data[self.start_date:self.end_date].loc[:, [
            self.asset]]
        name_column = "PandL_"+self.asset
        smaller_dataset[name_column] = -self.long_short * \
            self.quantity * (smaller_dataset[self.asset]-start_value)

        if self.stop_loss != None:
            for index, row in smaller_dataset.iterrows():
                if row[name_column] < self.stop_loss:
                    self.p_and_l_data = smaller_dataset[[
                        name_column]][smaller_dataset.index <= index]
                    self.p_and_l = self.p_and_l_data.iloc[-1, 0]
                    return
        self.p_and_l_data = smaller_dataset[[name_column]]
        self.p_and_l = self.p_and_l_data.iloc[-1, 0]
        self.win_trade = (self.p_and_l > 0)
        return
