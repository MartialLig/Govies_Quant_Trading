from _trade import Trade
from _long_short_trade import TradeLongShort
import pandas as pd
import plotly.express as px


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

    def compute_sharp_ratio(self):
        return

    def comute_total_return(self):
        return

    def comute_total_return_annualized(self):
        return

    def compute_percentage_win_trade(self):
        return

    def trades_sorted_by_rentability(self):
        return

    def plot_p_and_l(self):

        fig = px.line(self.p_and_l_dataset, x=self.p_and_l_dataset.index, y=self.p_and_l_dataset.columns,
                      title='Profit and Loss Over Time')
        fig.show()
        return
