

class Trade():
    def __init__(self, asset, quantity, long_short, start_date, end_date) -> None:
        self.asset = asset
        self.quantity = quantity
        self.long_short = long_short
        self.start_date = start_date
        self.end_date = end_date
        self.stop_loss = None  # pas coder
        self.take_profit = None  # pas coder
        self.p_and_l_data = None
        self.p_and_l = None
        pass

    def daily_Pand_L(self, data):
        # Cas sans stop_loss and take_profit
        start_value = data.loc[self.start_date, self.asset]
        smaller_dataset = data[self.start_date:self.end_date].loc[:, [
            self.asset]]
        name_column = "PandL_"+self.asset
        smaller_dataset[name_column] = -self.long_short * \
            self.quantity * (smaller_dataset[self.asset]-start_value)
        self.p_and_l_data = smaller_dataset[[name_column]]
        self.p_and_l = self.p_and_l_data.iloc[-1, 0]
        return
