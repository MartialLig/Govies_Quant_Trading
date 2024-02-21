import pandas as pd
import itertools


class DataManager():
    def __init__(self, path_data) -> None:
        self.data = pd.read_csv(path_data)
        self.data = self.set_datetime_index(self.data)
        self.data_by_country = self.generate_country_datasets(self.data)
        self.data_by_maturity = self.generate_maturity_datasets(self.data)
        self.rank_data = self.data.rank(axis=1)
        self.spread_yield = self.generate_spread_yield_datasets(self.data)
        pass

    def set_datetime_index(self, data):
        """
        Converts a specific column in the DataFrame to a datetime index and removes the original column.

        This function takes a DataFrame, converts the column named '100' to a datetime format, and then sets this column as the DataFrame's index. The original column ('100') is removed from the DataFrame, effectively updating the DataFrame to use datetime as its index for easier time-series manipulation.

        Parameters:
        - data (DataFrame): The input DataFrame where the column '100' contains date information in a format convertible to datetime.

        Returns:
        DataFrame: The modified DataFrame with 'date' as its datetime index and the original date column removed.
        """
        data["date"] = pd.to_datetime(data["100"])
        data.drop(columns=["100"], inplace=True)
        data.set_index('date', inplace=True)
        return data

    def generate_country_datasets(self, df):
        """
        Splits the original dataset into separate datasets for each country.

        Parameters:
        - df: The original DataFrame containing bond yield data.

        Returns:
        A dictionary where keys are country names and values are DataFrames specific to each country.
        """
        country_datasets = {}

        countries = set(col.split('_')[0] for col in df.columns if '_' in col)

        for country in countries:
            country_columns = [col for col in df.columns if col.startswith(
                country) or col == 'Date']
            country_datasets[country] = df[country_columns]

        return country_datasets

    def generate_maturity_datasets(self, df):
        """
        Splits the original dataset into separate datasets based on bond maturities.

        Parameters:
        - df: The original DataFrame containing bond yield data.

        Returns:
        A dictionary where keys are bond maturities (e.g., '5y', '10y', '30y') and values are DataFrames specific to each maturity.
        """
        maturity_datasets = {}

        maturities = set(col.split('_')[-1]
                         for col in df.columns if '_' in col)

        for maturity in maturities:
            maturity_columns = [col for col in df.columns if col.endswith(
                maturity) or col == 'Date']
            maturity_datasets[maturity] = df[maturity_columns]

        return maturity_datasets

    def generate_spread_yield_datasets(self, df):
        """
        Calculates the difference between each pair of yield columns in the dataset, creating a new DataFrame.

        This function takes the original DataFrame containing bond yield data and computes the spread between each possible pair of yield columns. The result is a new DataFrame where each column represents the difference (spread) between the yields of two specific bonds.

        Parameters:
        - df: The original DataFrame containing bond yield data. It is expected to have multiple columns, where each column represents a different bond yield.

        Returns:
        - A new DataFrame where each column name is a combination of the two original column names involved in the subtraction (e.g., 'BondA-BondB'), and each cell contains the difference between the corresponding yields in the original DataFrame.
        """

        column_names = df.columns
        column_pairs = list(itertools.combinations(
            column_names, 2))  # To create pairs

        df_diff = pd.DataFrame()
        for colA, colB in column_pairs:
            column_name = f"{colA}-{colB}"
            df_diff[column_name] = df[colA] - df[colB]
        return df_diff
