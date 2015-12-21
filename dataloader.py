import pandas as pd
import numpy as np

train_file = 'data/train.csv'
test_file = 'data/test.csv'
store_file = 'data/store.csv'

output_file = 'data/predictions.csv'

features = ['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday', 'StoreType', 'Assortment',
            'StateHoliday', 'DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear', 'CompetitionOpen', 'PromoOpen',
            'IsPromoMonth']


def load_train_data():
    types = {'CompetitionOpenSinceYear': np.dtype(int),
             'CompetitionOpenSinceMonth': np.dtype(int),
             'StateHoliday': np.dtype(str),
             'Promo2SinceWeek': np.dtype(int),
             'SchoolHoliday': np.dtype(float),
             'PromoInterval': np.dtype(str)}

    data = pd.read_csv(train_file, parse_dates=[2], dtype=types)
    store = pd.read_csv(store_file)

    data.fillna(1, inplace=True)

    data = data[data["Open"] != 0]
    data = data[data["Sales"] > 0]
    data = pd.merge(data, store, on='Store')

    return build_features(data)


def load_test_data():
    types = {'CompetitionOpenSinceYear': np.dtype(int),
             'CompetitionOpenSinceMonth': np.dtype(int),
             'StateHoliday': np.dtype(str),
             'Promo2SinceWeek': np.dtype(int),
             'SchoolHoliday': np.dtype(float),
             'PromoInterval': np.dtype(str)}

    data = pd.read_csv(test_file, parse_dates=[3], dtype=types)
    store = pd.read_csv(store_file)

    data.fillna(1, inplace=True)

    data = pd.merge(data, store, on='Store')

    return build_features(data)


def build_features(data):
    data.fillna(0, inplace=True)
    data.loc[data.Open.isnull(), 'Open'] = 1

    # Features 'Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday' are used directly
    features.extend([])

    # Features ['StoreType', 'Assortment', 'StateHoliday'] are mapped to 0,1,2,3,4 for values a,b,c,d accordingly
    features.extend(['StoreType', 'Assortment', 'StateHoliday'])
    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    # Features ['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'] are calculated from the Date feature
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear

    # Feature ['CompetitionOpen'] number of months since the competition opened
    features.append('CompetitionOpen')
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + (
                                    data.Month - data.CompetitionOpenSinceMonth)

    # ['PromoOpen'] Promo open time in months
    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + (
                              data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # ['IsPromoMonth'] - is 1 if the current month is a promo month in the store else 0
    month2str = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1

    return data
