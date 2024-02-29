import pandas as pd


class StratgiesComparetor():
    def __init__(self, list_straties, list_name_strategy) -> None:
        self.list_straties = list_straties
        self.list_name_strategy = list_name_strategy
        pass

    def compare_results_of_strategies(self):
        """
        Aggregates results from each strategy, renames the columns to match the strategy names,
        and concatenates them into a single DataFrame for comparison.

        Returns:
        - pd.DataFrame: A DataFrame where each column represents the aggregated results of a strategy,
          with column names corresponding to `list_name_strategy`.
        """
        list_of_agregate_perf_data = []
        for counter, strategy in enumerate(self.list_straties):
            new_data = strategy.aggregate_results()
            new_data.columns = [self.list_name_strategy[counter]]
            list_of_agregate_perf_data.append(new_data)
        return pd.concat(list_of_agregate_perf_data, axis=1)

    def plot_everything(self):
        """
        Plots the P&L for each strategy using the strategy's `plot_p_and_l` method.
        """
        for compteur, strategy in enumerate(self.list_straties):
            strategy.plot_p_and_l(self.list_name_strategy[compteur])

        return

    def combine_p_and_l_datasets(self):
        """
        Combines the P&L datasets from all strategies into a single DataFrame.

        Returns:
        - pd.DataFrame: A DataFrame where each column represents the P&L data from a strategy,
          with column names corresponding to `list_name_strategy`
        """
        combined_p_and_l = pd.DataFrame()
        for counter, strategy in enumerate(self.list_straties):
            if strategy.p_and_l_dataset is not None:
                combined_p_and_l[self.list_name_strategy[counter]
                                 ] = strategy.p_and_l_dataset["PandL"]
        return combined_p_and_l

    def calculate_correlations(self):
        """
        Calculates and returns the correlation matrix of the P&L datasets across all strategies.

        Returns:
        - pd.DataFrame: A correlation matrix of the P&L datasets from each strategy.
        """
        combined_p_and_l = self.combine_p_and_l_datasets()
        return combined_p_and_l.corr()
