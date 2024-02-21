import pandas as pd


class TradeLongShort():
    # stratégie qui consiste juste à quand deux indices se croisent, à être long un et short l'autre et
    def __init__(self, long, short, start_date) -> None:
        self.long = long
        self.short = short
        self.start_date = start_date
        pass

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
