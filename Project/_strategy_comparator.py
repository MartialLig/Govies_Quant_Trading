import pandas as pd


class StratgiesComparetor():
    def __init__(self, list_straties, list_name_strategy) -> None:
        self.list_straties = list_straties
        self.list_name_strategy = list_name_strategy
        pass

    def compare_results_of_strategies(self):
        list_of_agregate_perf_data = []
        for counter, strategy in enumerate(self.list_straties):
            new_data = strategy.aggregate_results()
            new_data.columns = [self.list_name_strategy[counter]]
            list_of_agregate_perf_data.append(new_data)
        return pd.concat(list_of_agregate_perf_data, axis=1)
