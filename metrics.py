import numpy as np
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale



def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)

def mean_smape(true, pred):
    raw_smape = smape(true, pred)
    masked_smape = np.ma.array(raw_smape, mask=np.isnan(raw_smape))
    return masked_smape.mean()

def normalize(data, method="MinMax", feature_range=(0, 1)):
    """
    normalize the data
    :param data: list of data
    :param method: support MinMax scaler or Z-Score scaler
    :param feature_range: use in MinMax scaler
    :return: normalized data(list), scaler
    """
    data = np.array(data)
    if len(data.shape) == 1 or data.shape[1] != 1:
        # reshape(-1, 1) --> reshape to a one column n rows matrix(-1 means not sure how many row)
        data = data.reshape(-1, 1)
    if method == "MinMax":
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == "Z-Score":
        scaler = StandardScaler()
    else:
        raise ValueError("only support MinMax scaler and Z-Score scaler")
    scaler.fit(data)
    # scaler transform apply to each column respectively
    # (which means that if we want to transform a 1-D data, we must reshape it to n x 1 matrix)
    return scaler.transform(data).reshape(-1), scaler


def denormalize(data, scaler):
    """
    denormalize data by scaler
    :param data:
    :param scaler:
    :return: denormalized data
    """
    data = np.array(data)
    if len(data.shape) == 1 or data.shape[1] != 1:
        data = data.reshape(-1, 1)
    # max, min, mean, variance are all store in scaler, so we need it to perform inverse transform
    return scaler.inverse_transform(data).reshape(-1)


def mean_error(y_true, y_pred):
    return K.mean(y_pred - y_true)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))