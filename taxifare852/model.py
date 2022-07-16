from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def get_model():

    model = RandomForestRegressor()

    return model


def get_grid_model(model_name):

    if model_name == 'random_forest':
        model = RandomForestRegressor()

    if model_name == 'linear_regression':
        model = LinearRegression()

    return model
