from _trade import Trade
from _long_short_trade import TradeLongShort



class TradeCreationLongShort():
    def __init__(self,list_of_orders) -> None:
        self.list_of_orders = list_of_orders
        self.list_of_trades = []
        pass

    def creation_list_of_trades(self):
        for index, row in self.list_of_orders.iterrows():
            for col in df.columns:
                if row[col] !=0:
                    quantity = 1
                    split_world = col.split("-")
                    if row[col] ==1:
                        long_asset = 
                        short_asset =
                    elif row[col] ==-11:
                    start_data = 
                    stop_date = 
                    trade = TradeLongShort()
        return 
    


class TradeCreation():
    def __init__(self,list_of_orders) -> None:
        self.list_of_orders = list_of_orders
        pass

    def creation_list_of_trades(self):
        return 