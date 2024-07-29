import pickle
import requests

url = 'http://127.0.0.1:5001/predict'

path = 'pipeline/support/aposteriori/'
columns = pickle.load(open(path + 'columns.pkl', 'rb'))


def get_user_input(show=False):
    default_check = input('Do you want to use the default check? [yes/no] ')

    if default_check.lower() == 'yes':
        import json
        with open('sample.json', 'r') as file:
            data = json.load(file)

    elif default_check.lower() == 'no':
        data = {}
        for col in columns:
            value = input(f'{col}: ')
            data[col] = float(value)

    if show:
        print(data)

    return data


data = get_user_input()

response = requests.post(url, json=data)
print('RESULT =>', response.json())