import pandas as pd
import os.path as path

from funcy import concat, count, repeat


# 大小関係がある文字列の特長量を数値に変換します。
def convert_to_number(data_frame):
    for feature, names in concat(zip(('Utilities',),
                                     repeat(('AllPub', 'NoSewr', 'NoSeWa', 'ELO'))),
                                 zip(('LandSlope',),
                                     repeat(('Gtl', 'Mod', 'Sev'))),
                                 zip(('ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'),
                                     repeat(('Ex', 'Gd', 'Ta', 'Fa', 'Po'))),
                                 zip(('BsmtExposure',),
                                     repeat(('Gd', 'Av', 'Mn', 'No'))),
                                 zip(('BsmtFinType1', 'BsmtFinType2'),
                                     repeat(('GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf'))),
                                 zip(('Functional',),
                                     repeat(('Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'))),
                                 zip(('GarageFinish',),
                                     repeat(('Fin', 'RFn', 'Unf'))),
                                 zip(('Fence',),
                                     repeat(('GdPrv', 'MnPrv', 'GdWo', 'MnWw')))):
        data_frame[feature] = data_frame[feature].map(dict(zip(names, count(len(names) - 1, -1)))).fillna(-1).astype('int')

    return data_frame


# 特徴量エンジニアリングで、特徴量を追加します。
def add_features(data_frame):
    data_frame['TotalSF'] = data_frame['TotalBsmtSF'] + data_frame['1stFlrSF'] + data_frame['2ndFlrSF']  # 3階建てはない？
    data_frame['SFPerRoom'] = data_frame['TotalSF'] / data_frame['TotRmsAbvGrd']

    return data_frame


# 訓練用のDataFrameを読み込みます。
def get_train_data_frame():
    return add_features(convert_to_number(pd.read_csv(path.join('..', 'input', 'house-prices-advanced-regression-techniques', 'train.csv'))))


# テスト用のDataFrameを読み込みます。
def get_test_data_frame():
    return add_features(convert_to_number(pd.read_csv(path.join('..', 'input', 'house-prices-advanced-regression-techniques', 'test.csv'))))


# カテゴリーの特長量を変換します。
def get_categorical_features(data_frame):
    return dict(map(lambda feature: (feature, dict(zip(data_frame[feature].factorize()[1], count()))),
                    ('MSZoning',
                     'Street',
                     'Alley',
                     'LotShape',
                     'LandContour',
                     'LotConfig',
                     'Neighborhood',
                     'Condition1',
                     'Condition2',
                     'BldgType',
                     'HouseStyle',
                     'RoofStyle',
                     'RoofMatl',
                     'Exterior1st',
                     'Exterior2nd',
                     'MasVnrType',
                     'Foundation',
                     'Heating',
                     'CentralAir',
                     'Electrical',
                     'GarageType',
                     'PavedDrive',
                     'MiscFeature',
                     'SaleType',
                     'SaleCondition')))


# 入力データを読み込みます。
def get_xs(data_frame, categorical_features):
    for feature, mapping in categorical_features.items():
        data_frame[feature] = data_frame[feature].map(mapping).fillna(-1).astype('category')

    return data_frame[['MSSubClass',
                       'MSZoning',
                       'LotFrontage',
                       'LotArea',
                       'Street',
                       'Alley',
                       'LotShape',
                       'LandContour',
                       'Utilities',
                       'LotConfig',
                       'LandSlope',
                       'Neighborhood',
                       'Condition1',
                       'Condition2',
                       'BldgType',
                       'HouseStyle',
                       'OverallQual',
                       'OverallCond',
                       'YearBuilt',
                       'YearRemodAdd',
                       'RoofStyle',
                       'RoofMatl',
                       'Exterior1st',
                       'Exterior2nd',
                       'MasVnrType',
                       'MasVnrArea',
                       'ExterQual',
                       'ExterCond',
                       'Foundation',
                       'BsmtQual',
                       'BsmtCond',
                       'BsmtExposure',
                       'BsmtFinType1',
                       'BsmtFinSF1',
                       'BsmtFinType2',
                       'BsmtFinSF2',
                       'BsmtUnfSF',
                       'TotalBsmtSF',
                       'Heating',
                       'HeatingQC',
                       'CentralAir',
                       'Electrical',
                       '1stFlrSF',
                       '2ndFlrSF',
                       'LowQualFinSF',
                       'GrLivArea',
                       'BsmtFullBath',
                       'BsmtHalfBath',
                       'FullBath',
                       'HalfBath',
                       'BedroomAbvGr',
                       'KitchenAbvGr',
                       'KitchenQual',
                       'TotRmsAbvGrd',
                       'Functional',
                       'Fireplaces',
                       'FireplaceQu',
                       'GarageType',
                       'GarageYrBlt',
                       'GarageFinish',
                       'GarageCars',
                       'GarageArea',
                       'GarageQual',
                       'GarageCond',
                       'PavedDrive',
                       'WoodDeckSF',
                       'OpenPorchSF',
                       'EnclosedPorch',
                       '3SsnPorch',
                       'ScreenPorch',
                       'PoolArea',
                       'PoolQC',
                       'Fence',
                       'MiscFeature',
                       'MiscVal',
                       'MoSold',
                       'YrSold',
                       'SaleType',
                       'SaleCondition',
                       'TotalSF',
                       'SFPerRoom']]


# 正解データを読み込みます。
def get_ys(data_frame):
    return data_frame['SalePrice']
