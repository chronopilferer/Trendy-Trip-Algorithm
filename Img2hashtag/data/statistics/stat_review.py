import pandas as pd 

if __name__ == '__main__':
    data = pd.read_csv('statistics.csv')

    import pprint

    pprint.pprint(data)