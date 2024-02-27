
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class OptimisationTimeRegressison():

    def __init__(self) -> None:
        pass

    def find_optimal_value(self, dataset, set_time_to_compare, target_column):
        results_detailed = {}
        quick_results_data = []
        for train_duration, test_duration in set_time_to_compare:
            liste_data = self.create_all_datasets(
                dataset, train_duration, test_duration)
            result_summary = self.run_regressions_and_save_metrics(
                liste_data, target_column)
            results_detailed[(train_duration, test_duration)] = result_summary

            # Filter the DataFrame for positive and negative R2 Test values separately
            positive_r2_test = result_summary[result_summary["Test_R2"]
                                              > 0]["Test_R2"]
            negative_r2_test = result_summary[result_summary["Test_R2"]
                                              < 0]["Test_R2"]

            # Calculate the mean for both positive and negative R2 Test values
            mean_positive_r2_test = positive_r2_test.mean()
            mean_negative_r2_test = negative_r2_test.mean()

            # Drop 'start_date_test' column if it exists and calculate the mean for other columns
            mean_values = result_summary.drop(
                columns=['start_date_test'], errors='ignore').mean().to_dict()

            # Add the mean positive and negative R2 Test values to the mean_values dictionary
            mean_values['mean_positive_r2_test'] = mean_positive_r2_test
            mean_values['mean_negative_r2_test'] = mean_negative_r2_test

            quick_results_data.append(
                ((train_duration, test_duration), mean_values))

            '''
            mean_values = result_summary.drop(
                columns=['start_date_test'], errors='ignore').mean().to_dict()
            quick_results_data.append(
                ((train_duration, test_duration), mean_values))'''

        quick_results_index = pd.MultiIndex.from_tuples(
            [x[0] for x in quick_results_data], names=['train_duration', 'test_duration'])
        quick_results = pd.DataFrame(
            [x[1] for x in quick_results_data], index=quick_results_index)
        return quick_results, results_detailed

    def run_regressions_and_save_metrics(self, list_of_datasets,  target_column):

        metrics = []

        for train_set, test_set in list_of_datasets:
            X_train = train_set.drop(columns=[target_column])
            y_train = train_set[target_column]
            X_test = test_set.drop(columns=[target_column])
            y_test = test_set[target_column]

            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions for the training set
            train_predictions = model.predict(X_train)
            train_r2 = r2_score(y_train, train_predictions)
            train_rmse = np.sqrt(mean_squared_error(
                y_train, train_predictions))
            train_mae = mean_absolute_error(
                y_train, train_predictions)  # MAE for the training set

            # Predictions for the test set
            test_predictions = model.predict(X_test)
            test_r2 = r2_score(y_test, test_predictions)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            test_mae = mean_absolute_error(y_test, test_predictions)

            metrics.append({
                'Train_R2': train_r2,
                'Train_RMSE': train_rmse,
                'Train_MAE': train_mae,
                'Test_R2': test_r2,
                'Test_RMSE': test_rmse,
                'Test_MAE': test_mae,
                'start_date_test': test_set.index[-1]
            })
        metrics_df = pd.DataFrame(metrics)

        return metrics_df

    def create_all_datasets(self, df, train_duration, test_duration):
        limit_date = df.index[0]
        list_of_datasets = []
        while limit_date < pd.to_datetime("2019-07-01"):
            train, test = self.train_test_split_time_series(
                df, train_duration, test_duration)

            list_of_datasets.append([train, test])
            last_date_to_train_next = test.index[-1]
            firs_date_of_next_dataset = last_date_to_train_next - \
                pd.DateOffset(months=train_duration)
            df = df[df.index >= firs_date_of_next_dataset]
            limit_date = last_date_to_train_next

        return list_of_datasets

    def train_test_split_time_series(self, df, train_duration, test_duration):

        end_train_date = df.index[0] + pd.DateOffset(months=train_duration)
        print(end_train_date)

        start_test_date = end_train_date

        end_test_date = start_test_date + pd.DateOffset(months=test_duration)

        train_set = df[df.index < end_train_date]
        test_set = df[(df.index >= start_test_date)
                      & (df.index < end_test_date)]

        return train_set, test_set

    def run_regression_and_focus(self, dataset, train_duration, test_duration, target_column):
        liste_data = self.create_all_datasets(
            dataset, train_duration, test_duration)
        metrics = []

        # Prepare the figure layout for plotting
        # Doubling the number of plots to accommodate error plots
        fig, axs = plt.subplots(len(liste_data)*4, 1,
                                figsize=(10, len(liste_data)*16))
        fig.subplots_adjust(hspace=0.5)

        for i, (train_set, test_set) in enumerate(liste_data):
            X_train = train_set.drop(columns=[target_column])
            y_train = train_set[target_column]
            X_test = test_set.drop(columns=[target_column])
            y_test = test_set[target_column]

            model = Ridge()
            model.fit(X_train, y_train)

            # Predictions for the training set
            train_predictions = model.predict(X_train)
            train_r2 = r2_score(y_train, train_predictions)
            train_rmse = np.sqrt(mean_squared_error(
                y_train, train_predictions))
            train_mae = mean_absolute_error(y_train, train_predictions)

            # Predictions for the test set
            test_predictions = model.predict(X_test)
            test_r2 = r2_score(y_test, test_predictions)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            test_mae = mean_absolute_error(y_test, test_predictions)

            # Assuming the index is datetime
            start_date_test = test_set.index[0]
            metrics.append({
                'Train_R2': train_r2,
                'Train_RMSE': train_rmse,
                'Train_MAE': train_mae,
                'Test_R2': test_r2,
                'Test_RMSE': test_rmse,
                'Test_MAE': test_mae,
                'start_date_test': start_date_test
            })

            # Plotting errors for training predictions
            train_errors = y_train - train_predictions
            axs[i*4].scatter(train_predictions, train_errors)
            axs[i*4].hlines(y=0, xmin=train_predictions.min(),
                            xmax=train_predictions.max(), colors='k', linestyles='dashed')
            axs[i*4].set_title(
                f'Training Set Errors vs Predictions (Start date: {start_date_test})')
            axs[i*4].set_xlabel('Predicted')
            axs[i*4].set_ylabel('Errors')

            # Plotting errors for test predictions
            test_errors = y_test - test_predictions
            axs[i*4+2].scatter(test_predictions, test_errors)
            axs[i*4+2].hlines(y=0, xmin=test_predictions.min(),
                              xmax=test_predictions.max(), colors='k', linestyles='dashed')
            axs[i*4+2].set_title(
                f'Testing Set Errors vs Predictions (Start date: {start_date_test})')
            axs[i*4+2].set_xlabel('Predicted')
            axs[i*4+2].set_ylabel('Errors')

        plt.show()

        metrics_df = pd.DataFrame(metrics)
        return metrics_df


class AssetSelectioByModel:
    def __init__(self, data, start_date, end_date, model):
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.model = model
        self.results_df = None
        self.df = None

    def preprocess_data(self):
        new_data = self.data.diff(5)
        new_data = new_data.dropna(axis=0)
        self.df = new_data[(new_data.index >= self.start_date)
                           & (new_data.index < self.end_date)]

    def train_and_evaluate(self):
        if self.df is None:
            self.preprocess_data()

        model = self.model
        results = []

        for column in self.df.columns:
            y = self.df[column].shift(-5)
            X = self.df.drop(columns=[column]).iloc[:-5]
            y = y.dropna()
            y.to_numpy()
            X.to_numpy()
            if not X.empty and len(X) == len(y):
                model.fit(X, y)
                predictions = model.predict(X)
                mse = mean_squared_error(y, predictions)
                rmse = np.sqrt(mse)
                result = {
                    'column': column,
                    'score': model.score(X, y),
                    'MSE': mse,
                    'RMSE': rmse
                }
                results.append(result)

        self.results_df = pd.DataFrame(results)
        return self.results_df.sort_values("score", ascending=False)
