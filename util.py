import pickle
import json
import numpy as np
import math

__state = None
__data_columns = None
__model = None

def get_covid19_pred(state,confirmed,year,day):
    try:
        loc_index = __data_columns.index(state.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))

    x[0] = confirmed
    x[1] = year
    x[2] = day
    if loc_index>=0:
        x[loc_index] = 1

    return math.floor(round(__model.predict([x])[0],2))


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __state

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __state = __data_columns[3:]  # first 3 columns are sqft, bath, bhk

    global __model
    if __model is None:
        with open('./artifacts/covid_19.pkl', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_state_names():
    return __state

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_state_names())
    print(get_covid19_pred('kerala',1000, 2021, 10))
    print(get_covid19_pred('andaman and nicobar islands', 156, 2021, 23))
    print(get_covid19_pred('bihar', 567, 2020, 22)) # other location
    print(get_covid19_pred('assam', 1450, 2021, 13))  # other location