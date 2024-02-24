import pandas as pd
from _trade import Trade


class TradeLongShort():
    # stratégie qui consiste juste à quand deux indices se croisent, à être long un et short l'autre et
    def __init__(self, long_asset, short_asset, quantity, start_date, end_date) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.long_asset = long_asset
        self.short_asset = short_asset
        self.long_trade = Trade(long_asset, quantity, 1,
                                self.start_date, self.end_date)
        self.short_trade = Trade(
            short_asset, quantity, -1, self.start_date, self.end_date)
        self.p_and_l_data = None
        self.p_and_l_data_detailed = None
        self.p_and_l = None

        pass

    def daily_Pand_L(self, data):
        self.long_trade.daily_Pand_L(data)
        self.short_trade.daily_Pand_L(data)
        self.p_and_l_data_detailed = pd.merge(
            self.long_trade.p_and_l_data, self.short_trade.p_and_l_data, left_index=True, right_index=True)
        self.p_and_l_data_detailed["p_and_l"] = self.p_and_l_data_detailed.iloc[:,
                                                                                0]+self.p_and_l_data_detailed.iloc[:, 1]
        new_name = "p_and_l_"+self.long_asset+"-"+self.short_asset
        self.p_and_l_data = self.p_and_l_data_detailed[[
            'p_and_l']].rename(columns={'p_and_l': new_name})
        self.p_and_l = self.p_and_l_data.iloc[-1, 0]
        return


'''
Old version
    def delta_yield_days_to_hold_range(self, range_day_to_hold, dataset):
        delta_yield_days_list = []
        for business_day_to_hold in range_day_to_hold:
            delta_yield_days_list.append(
                self.delta_yield(business_day_to_hold, dataset))
        return delta_yield_days_list

    def delta_yield(self, business_day_to_hold, dataset):
        starting_yield_long = dataset[self.long].loc[self.start_date]
        starting_yield_short = dataset[self.short].loc[self.start_date]

        position_date = dataset.index.get_loc(self.start_dat)

        ending_yield_long = dataset[self.long].iloc[position_date +
                                                    business_day_to_hold]
        ending_yield_short = dataset[self.short].iloc[position_date +
                                                      business_day_to_hold]

        difference_yield_long = starting_yield_long - ending_yield_long
        difference_yield_short = ending_yield_short - starting_yield_short
        difference = difference_yield_long + difference_yield_short

        return difference
'''
