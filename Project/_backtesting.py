from _trade import Trade
from _long_short_trade import TradeLongShort
import pandas as pd
import plotly.express as px
import numpy as np


class Backtest():
    def __init__(self, list_of_trades, data) -> None:
        self.list_of_trades = list_of_trades
        self.data = data
        self.p_and_l_detailed_dataset = None
        self.p_and_l_dataset = None
        pass

    def gather_all_trades(self):
        """
        Gathers and merges the profit and loss data from all trades in the `list_of_trades`
        into a single DataFrame.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame containing the merged profit and loss data for all trades.
        """
        data = self.data[[]]
        for compteur, trade in enumerate(self.list_of_trades):
            trade.daily_Pand_L(self.data)
            name_col = "id"+str(compteur) + "_" + trade.p_and_l_data.columns[0]
            data_pand_l_to_merge = trade.p_and_l_data
            data_pand_l_to_merge.columns = [name_col]
            data = pd.merge(data, data_pand_l_to_merge, "left",
                            left_index=True, right_index=True)

        data = self.complete_dataset(data)

        data['PandL'] = data.sum(axis=1)
        cols = ['PandL'] + [col for col in data.columns if col != 'PandL']
        data = data[cols]

        self.p_and_l_detailed_dataset = data
        self.p_and_l_dataset = data[['PandL']]
        return

    def complete_dataset(self, data):
        """
        Completes the dataset by filling in missing values with the first non-NaN
        value before and the last value after NaNs.

        Parameters:
        data : pandas.DataFrame :The DataFrame to be completed.

        Returns:
        pandas.DataFrame :The completed DataFrame with missing values filled.
        """
        def remplir_colonne(colonne):
            if pd.isna(colonne.iloc[0]):
                premiere_valeur_non_nan = colonne.first_valid_index()
                position_premier_valeur = colonne.index.get_loc(
                    premiere_valeur_non_nan)
                colonne[0:position_premier_valeur] = colonne[0:position_premier_valeur].fillna(
                    0)
                derniere_valeur_non_nan = colonne.last_valid_index()
                position = colonne.index.get_loc(derniere_valeur_non_nan)
                if position < len(colonne) - 1:
                    derniere_valeur = colonne[derniere_valeur_non_nan]
                    colonne[position +
                            1:] = colonne[position + 1:].fillna(derniere_valeur)
            else:
                derniere_valeur_non_nan = colonne.last_valid_index()
                position = colonne.index.get_loc(derniere_valeur_non_nan)
                if position < len(colonne) - 1:
                    derniere_valeur = colonne[derniere_valeur_non_nan]
                    colonne[position +
                            1:] = colonne[position + 1:].fillna(derniere_valeur)
            return colonne

        for colonne in data.columns:
            data[colonne] = remplir_colonne(data[colonne])

        return data

    def data_with_adjusted_index(self):
        start_index = self.p_and_l_dataset[self.p_and_l_dataset["PandL"] != 0].first_valid_index(
        )

        diff = self.p_and_l_dataset["PandL"] .diff().ne(0)
        end_index = diff[diff].last_valid_index()

        self.p_and_l_dataset = self.p_and_l_dataset.loc[start_index:end_index]

        return

    def aggregate_results(self):
        sharp_ratio = self.compute_sharp_ratio()
        total_return = self.compute_total_return()
        percentage_win_trade = self.compute_percentage_win_trade()
        worst_drawdown, date_of_worst_drawdown = self.compute_worst_drawdown()
        number_of_trade = self.number_of_trade()
        annual_performance = self.performance_by_year()

        results = {
            "Sharp Ratio": [sharp_ratio],
            "Total Return": [total_return],
            "Percentage Win Trade": [percentage_win_trade],
            "Number of Trades": [number_of_trade],
            "Worst Drawdown": [worst_drawdown],
            "Date of Worst Drawdown": [date_of_worst_drawdown]
        }

        results_df = pd.DataFrame(results)

        for year in annual_performance.index:
            results_df[f'Annual Performance {year}'] = annual_performance.loc[year].values[0]

        return results_df.T

    def compute_sharp_ratio(self):
        return round(self.p_and_l_dataset["PandL"].diff(
            1).mean()/self.p_and_l_dataset["PandL"].diff(1).std()*np.sqrt(252), 2)

    def compute_total_return(self):
        return self.p_and_l_dataset.iloc[-1, 0]-self.p_and_l_dataset.iloc[0, 0]

    def compute_percentage_win_trade(self):
        compteur = 0
        for trade in self.list_of_trades:
            if trade.win_trade == True:
                compteur += 1
        return round(compteur/len(self.list_of_trades), 3)

    def compute_worst_drawdown(self):
        p_and_l_dataset = self.p_and_l_dataset
        p_and_l_dataset['max_to_date'] = p_and_l_dataset['PandL'].cummax()

        p_and_l_dataset['drawdown'] = p_and_l_dataset['PandL'] - \
            p_and_l_dataset['max_to_date']

        worst_drawdown = self.p_and_l_dataset['drawdown'].min()
        date_of_worst_drawdown = p_and_l_dataset[p_and_l_dataset['drawdown']
                                                 == worst_drawdown].index[0]

        return worst_drawdown, date_of_worst_drawdown

    def performance_by_year(self):
        anual_performance = self.p_and_l_dataset.resample(
            'Y').last() - self.p_and_l_dataset.resample('Y').first()
        anual_performance.index = anual_performance.index.year
        return anual_performance

    def number_of_trade(self):
        return len(self.list_of_trades)

    def trades_sorted_by_rentability(self):
        return

    def plot_p_and_l(self, title=None):
        if title == None:
            title = 'Profit and Loss Over Time'

        fig = px.line(self.p_and_l_dataset, x=self.p_and_l_dataset.index, y=self.p_and_l_dataset.columns,
                      title=title)
        fig.update_layout(title_text=title, title_x=0.5)
        fig.show()
        return
