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

    def plot_everything(self):
        for compteur, strategy in enumerate(self.list_straties):
            strategy.plot_p_and_l(self.list_name_strategy[compteur])

        return

    def combine_p_and_l_datasets(self):
        combined_p_and_l = pd.DataFrame()
        for counter, strategy in enumerate(self.list_straties):
            if strategy.p_and_l_dataset is not None:
                combined_p_and_l[self.list_name_strategy[counter]
                                 ] = strategy.p_and_l_dataset["PandL"]
        return combined_p_and_l

    def calculate_correlations(self):
        combined_p_and_l = self.combine_p_and_l_datasets()
        return combined_p_and_l.corr()
