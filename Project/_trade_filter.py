

class TradeFilter:
    def __init__(self, list_of_trades, by_realised_vol) -> None:
        self.list_of_trades = list_of_trades
        self.by_realised_vol = by_realised_vol
        pass

    def filter_by_vol(self, data_vol):
        """
        Filters the trades based on realized volatility.

        Parameters:
            data_vol (DataFrame): A DataFrame indexed by date, containing realized volatility of an asset.

        Returns:
            list: A list of Trade objects that meet the volatility filtering criteria.
        """
        new_list_of_trades = []
        for trade in self.list_of_trades:
            if data_vol.loc[trade.start_date] > self.by_realised_vol:
                new_list_of_trades.append(trade)

        return new_list_of_trades
