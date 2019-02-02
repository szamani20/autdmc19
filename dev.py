import math
from collections import Counter
from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time
from sklearn.neighbors import KNeighborsClassifier

from scipy import stats

from constants import *

fix, ax = plt.subplots()


class CSVFile:
    def __init__(self, path):
        self.path = path
        self.data_frame = None

    def read_csv(self, usecols=None, dtype=None, converted=None,
                 skiprows=None, nrows=None, infer_datetime_format=False):
        self.data_frame = pd.read_csv(self.path, nrows=nrows, dtype=DTYPE, usecols=usecols)

    def select_column_by_number(self, column_number):
        return self.data_frame.iloc[:, column_number]

    # column_range:  (2, 4)
    def select_column_range_by_number(self, column_range):
        return self.data_frame.iloc[:, column_range[0]:column_range[1]]

    def select_row_by_number(self, row_number):
        return self.data_frame.iloc[row_number]

    # row_range:   (2, 4)
    def select_row_range_by_number(self, row_range):
        return self.data_frame.iloc[row_range[0]:row_range[1]]

    def select_column_by_name(self, column_name):
        return self.data_frame.loc[:, [column_name]]

    # column_range:   ('AL', 'Price')
    def select_column_range_by_name(self, column_range):
        return self.data_frame.loc[:, column_range[0]:column_range[1]]

    def select_rows_by_year_filter(self, year):
        return self.data_frame.loc[self.data_frame['Log_Date'].str.startswith(year)]


class Utils:
    @staticmethod
    def time_to_seconds(human_readable_time):
        print(human_readable_time)

    @staticmethod
    def extract_month_from_jalali_date(jalali_dates):
        converter = lambda x: int(x.split('/')[1])
        return jalali_dates.map(converter)

    @staticmethod
    def airline_vs_flight_counts():
        train_data_frame = CSVFile(TRAIN_DATA_PATH)
        train_data_frame.read_csv()

        airline_series = train_data_frame.select_column_by_number(2)
        airline_count = airline_series.value_counts()
        all_flights = sum(airline_count.values)
        for i, v in enumerate(airline_count.values):
            ax.text(v + 5, i - 0.25, str((v / all_flights) * 100)[:2].replace('.', ''))
        airline_count.plot.barh()
        plt.xlabel('Fly counts')
        plt.ylabel('Airline')
        plt.show()

    @staticmethod
    def month_vs_flight_counts():
        train_data_frame = CSVFile(TRAIN_DATA_PATH)
        train_data_frame.read_csv()

        jalali_dates = train_data_frame.select_column_by_number(1)
        month_series = Utils.extract_month_from_jalali_date(jalali_dates.copy())
        month_count = month_series.value_counts()
        all_flights = sum(month_count.values)
        for i, v in enumerate(month_count.values):
            ax.text(v + 5, i - 0.2, str((v / all_flights) * 100)[:2].replace('.', ''))
        month_count.plot.barh()
        plt.xlabel('Fly counts')
        plt.ylabel('Month')
        plt.show()

    @staticmethod
    def from_vs_flight_counts():
        train_data_frame = CSVFile(TRAIN_DATA_PATH)
        train_data_frame.read_csv()

        from_series = train_data_frame.select_column_by_number(3)
        from_count = from_series.value_counts()
        for i, v in enumerate(from_count.values):
            ax.text(v + 5, i - 0.2, str(v / 1000).split('.')[0])
        from_count.plot.barh(width=0.4)
        plt.xlabel('Fly counts (x1000)')
        plt.ylabel('From Airport')
        plt.show()

    @staticmethod
    def to_vs_flight_counts():
        train_data_frame = CSVFile(TRAIN_DATA_PATH)
        train_data_frame.read_csv()

        to_series = train_data_frame.select_column_by_number(4)
        to_count = to_series.value_counts()
        for i, v in enumerate(to_count.values):
            ax.text(v + 5, i - 0.2, str(v / 1000).split('.')[0])
        to_count.plot.barh(width=0.4)
        plt.xlabel('Fly counts (x1000)')
        plt.ylabel('To Airport')
        plt.show()

    @staticmethod
    def jalali_to_days_converter(date):
        month = int(date.split('/')[1])
        day = int(date.split('/')[2])
        if month <= 6:
            return (month - 1) * 31 + day
        else:
            return 186 + (month - 7) * 30 + day

    @staticmethod
    def convert_jalali_to_day_of_year(column):
        return column.map(lambda x: Utils.jalali_to_days_converter(x))

    @staticmethod
    def extract_train_data_two_years(which_usecols='flight_and_price'):
        train_data_frame = CSVFile(TRAIN_DATA_PATH)
        if which_usecols == 'flight_and_price':
            train_data_frame.read_csv(usecols=['Log_Date', 'FROM', 'TO', 'Price'])
        else:
            train_data_frame.read_csv(usecols=['Log_Date', 'Departure_Date'])

        train_data_95 = train_data_frame.select_rows_by_year_filter('1395')
        doy_95 = Utils.convert_jalali_to_day_of_year(train_data_95.iloc[:, 0])
        train_data_95.loc[:, 'Log_Date'] = doy_95

        train_data_96 = train_data_frame.select_rows_by_year_filter('1396')
        doy_96 = Utils.convert_jalali_to_day_of_year(train_data_96.iloc[:, 0])
        train_data_96.loc[:, 'Log_Date'] = doy_96

        if which_usecols == 'flight_and_departure':
            doy_departure_95 = Utils.convert_jalali_to_day_of_year(train_data_95.iloc[:, 1])
            doy_departure_96 = Utils.convert_jalali_to_day_of_year(train_data_96.iloc[:, 1])
            train_data_95.loc[:, 'Departure_Date'] = doy_departure_95
            train_data_96.loc[:, 'Departure_Date'] = doy_departure_96

        return train_data_95, train_data_96

    @staticmethod
    def plot_two_charts(xlabel, ylabel, flights_95, flights_96):
        plt.plot(flights_95, label='95')
        plt.plot(flights_96, label='96')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        plt.show()

    @staticmethod
    def plot_flights_per_day():
        train_data_95, train_data_96 = Utils.extract_train_data_two_years()

        flights_95 = train_data_95.groupby('Log_Date').count().iloc[:, 0]
        flights_96 = train_data_96.groupby('Log_Date').count().iloc[:, 0]

        Utils.plot_two_charts('Day of Year', 'Number of Flights Registered',
                              flights_95, flights_96)

    @staticmethod
    def plot_total_price_per_day():
        train_data_95, train_data_96 = Utils.extract_train_data_two_years()

        flights_95 = train_data_95.groupby('Log_Date')['Price'].sum()
        flights_96 = train_data_96.groupby('Log_Date')['Price'].sum()

        Utils.plot_two_charts('Day of Year', 'Total Price',
                              flights_95, flights_96)

    @staticmethod
    def plot_average_price_per_day():
        train_data_95, train_data_96 = Utils.extract_train_data_two_years()

        flights_95 = train_data_95.groupby('Log_Date')['Price'].sum() / train_data_95.groupby('Log_Date').count().iloc[
                                                                        :, 2]
        flights_96 = train_data_96.groupby('Log_Date')['Price'].sum() / train_data_96.groupby('Log_Date').count().iloc[
                                                                        :, 2]

        Utils.plot_two_charts('Day of Year', 'Average Price Per Flight',
                              flights_95, flights_96)

    @staticmethod
    def extract_flight_log_and_departure_mode():
        train_data_95, train_data_96 = Utils.extract_train_data_two_years('flight_and_departure')

        number_of_registered_flights = train_data_95.groupby('Log_Date').count()
        list_of_departure_dates = train_data_95.groupby('Log_Date')['Departure_Date'].apply(list)
        mode_and_count_of_departure_dates = train_data_95.groupby('Log_Date').agg(
            lambda x: (stats.mode(x)[0][0], stats.mode(x)[1][0]))

        number_of_registered_flights = number_of_registered_flights.assign(
            Departure_Mode_and_Count=mode_and_count_of_departure_dates.iloc[:, 0])

        number_of_registered_flights.rename(columns={'Departure_Date': 'Departure_Count'}, inplace=True)
        number_of_registered_flights.to_csv('log_date-number_of_departures-departures_mode_and_count_95.csv')

    @staticmethod
    def extract_flight_log_and_departure_counts():
        train_data_95, train_data_96 = Utils.extract_train_data_two_years('flight_and_departure')

        number_of_registered_flights = train_data_95.groupby('Log_Date').count()
        departure_date_counts_95 = train_data_95.groupby('Log_Date')['Departure_Date'].value_counts()
        # departure_date_counts_95 = departure_date_counts_95[departure_date_counts_95 > 10]

        departure_date_counts_95.to_csv('departure_date_counts_95.csv')

    @staticmethod
    def fibonacci_model():
        test_data = pd.read_csv('../test.csv', usecols=['Log_Date', 'From', 'To'])
        original_test_data = test_data.copy()
        doy_test = Utils.convert_jalali_to_day_of_year(test_data.iloc[:, 0])
        test_data.loc[:, 'Log_Date'] = doy_test

        t95 = pd.read_csv('test_shaped/95.csv', dtype={'FROM': np.uint16, 'TO': np.uint16})
        t96 = pd.read_csv('test_shaped/96.csv', dtype={'FROM': np.uint16, 'TO': np.uint16})

        sales_list = []

        for _, row in test_data.iterrows():
            Log_Date, FROM, TO = row['Log_Date'], row['From'], row['To']
            count_95 = t95[(t95['Log_Date'] == Log_Date) & (t95['FROM'] == FROM) & (t95['TO'] == TO)]
            count_96 = t96[(t96['Log_Date'] == Log_Date) & (t96['FROM'] == FROM) & (t96['TO'] == TO)]
            window95 = t95[(t95['Log_Date'] > Log_Date - 10) & (t95['Log_Date'] < Log_Date + 10)
                           & (t95['FROM'] == FROM) & (t95['TO'] == TO)]
            window96 = t96[(t96['Log_Date'] > Log_Date - 10) & (t96['Log_Date'] < Log_Date + 10)
                           & (t96['FROM'] == FROM) & (t96['TO'] == TO)]

            if count_95.empty and count_96.empty:
                if window95.empty and window96.empty:
                    sales_list.append(0)
                elif window95.empty and not window96.empty:
                    sales_list.append(math.floor(window96['Price'].sum() / window96.shape[0]))
                elif window96.empty and not window95.empty:
                    sales_list.append(math.floor(window95['Price'].sum() / window95.shape[0]))
                else:
                    sales_list.append(math.floor((math.floor(window96['Price'].sum() / window96.shape[0]) +
                                      math.floor(window95['Price'].sum() / window95.shape[0])) / 2))
            elif count_95.empty and not count_96.empty:
                c96 = count_96.iloc[0, 3]
                if not window95.empty:
                    sales_list.append(c96 + math.floor(window95['Price'].sum() / window95.shape[0]))
                else:
                    sales_list.append(c96)
            elif count_96.empty and not count_95.empty:
                c95 = count_95.iloc[0, 3]
                if not window96.empty:
                    sales_list.append(c95 + math.floor(window96['Price'].sum() / window96.shape[0]))
                else:
                    sales_list.append(c95)
            else:
                sales_list.append(count_95.iloc[0, 3] + count_96.iloc[0, 3])

        sales = pd.Series(sales_list)

        original_test_data = original_test_data.assign(Sales=sales.values)
        original_test_data.to_csv('result/resultb.csv', index=False)

    @staticmethod
    def save_grouped_by_log_from_to_doy():
        train_data_95, train_data_96 = Utils.extract_train_data_two_years()

        test_shape_data_95 = train_data_95.groupby(['Log_Date', 'FROM', 'TO']).count()
        test_shape_data_96 = train_data_96.groupby(['Log_Date', 'FROM', 'TO']).count()

        # test_shape_data_95 = test_shape_data_95.reindex(columns=['Log_Date', 'FROM', 'TO', 'Price'])
        # test_shape_data_96 = test_shape_data_96.reindex(columns=['Log_Date', 'FROM', 'TO', 'Price'])

        # test_shape_data_95 = test_shape_data_95.astype({'FROM': np.int, 'TO': np.int})
        # test_shape_data_96 = test_shape_data_96.astype({'FROM': np.int, 'TO': np.int})

        test_shape_data_95.to_csv('test_shaped/95.csv')
        test_shape_data_96.to_csv('test_shaped/96.csv')

    @staticmethod
    def save_grouped_by_log_from_to_jalali():
        train_data_frame = CSVFile(TRAIN_DATA_PATH)
        train_data_frame.read_csv(usecols=['Log_Date', 'FROM', 'TO', 'Price'])

        train_data_95 = train_data_frame.select_rows_by_year_filter('1395')
        train_data_96 = train_data_frame.select_rows_by_year_filter('1396')

        train_data_95 = train_data_95.groupby(['Log_Date', 'FROM', 'TO']).count()
        train_data_96 = train_data_96.groupby(['Log_Date', 'FROM', 'TO']).count()

        # pprint(train_data_95[train_data_95.apply(lambda x: x.isnull().all(), axis=1)])

        # train_data_95.dropna(inplace=True)
        # train_data_96.dropna(inplace=True)

        # train_data_95 = train_data_95[~train_data_95.isin([np.nan, np.inf, -np.inf]).any(1)]
        # train_data_96 = train_data_96[~train_data_96.isin([np.nan, np.inf, -np.inf]).any(1)]

        # train_data_95 = train_data_95.fillna(0).astype(np.uint16)
        # train_data_96 = train_data_96.fillna(0).astype(np.uint16)

        # train_data_95 = train_data_95.reindex(columns=['Log_Date', 'FROM', 'TO', 'Price'])
        # train_data_96 = train_data_96.reindex(columns=['Log_Date', 'FROM', 'TO', 'Price'])

        # train_data_95 = train_data_95.astype({'FROM': np.uint16, 'TO': np.uint16})
        # train_data_96 = train_data_96.astype({'FROM': np.uint16, 'TO': np.uint16})

        train_data_95.to_csv('test_shaped/95_jalali.csv')
        train_data_96.to_csv('test_shaped/96_jalali.csv')


class LearningModel:
    def __init__(self, train_data, test_data):
        pass


if __name__ == '__main__':
    # Utils.plot_average_price_per_day()
    # Utils.plot_total_price_per_day()
    # Utils.plot_flights_per_day()
    # Utils.to_vs_flight_counts()
    # Utils.from_vs_flight_counts()
    # Utils.month_vs_flight_counts()
    # Utils.airline_vs_flight_counts()
    # Utils.extract_flight_log_and_departure_mode()
    # Utils.extract_flight_log_and_departure_counts()
    # Utils.save_grouped_by_log_from_to_doy()
    # Utils.save_grouped_by_log_from_to_jalali()

    Utils.fibonacci_model()

    # learning_model = LearningModel()
