import pandas as pd
# print(pd.__version__)

URL = "https://www.multipasko.pl/wyniki-csv.php?f=minilotto-sortowane"
data = pd.read_csv(URL, delimiter=';')
data = data.tail(100)
data.to_csv('all_data.csv')
# print(data.tail(5))
data = data.tail(100)
# data_L1 = data.drop(columns=['Numer', 'Miesiac', 'Rok', 'L2', 'L3', 'L4', 'L5'])
# data_L1 = data_L1.rename(columns={'Dzien': 'X', 'L1': 'y'})
# data_L1 = data_L1.rename(columns={'L1': 'y'})
data_y_L1 = data.drop(columns=['Numer', 'Dzien', 'Miesiac', 'Rok', 'L2', 'L3', 'L4', 'L5'])
data_y_L2 = data.drop(columns=['Numer', 'Dzien', 'Miesiac', 'Rok', 'L1', 'L3', 'L4', 'L5'])
data_y_L3 = data.drop(columns=['Numer', 'Dzien', 'Miesiac', 'Rok', 'L1', 'L2', 'L4', 'L5'])
data_y_L4 = data.drop(columns=['Numer', 'Dzien', 'Miesiac', 'Rok', 'L1', 'L2', 'L3', 'L5'])
data_y_L5 = data.drop(columns=['Numer', 'Dzien', 'Miesiac', 'Rok', 'L1', 'L2', 'L3', 'L4'])
data_y_L15 = data.drop(columns=['Numer', 'Dzien', 'Miesiac', 'Rok'])

# data_y = data.drop(columns=['Numer', 'Dzien', 'Miesiac', 'Rok'])

data_X = data.drop(columns=['Dzien', 'Miesiac', 'Rok', 'L1', 'L2', 'L3', 'L4', 'L5'])

# print(f"{data_L1.tail(5)}")
data_y_L1.to_csv('results_y_L1.csv', index=False, header=False)
data_y_L2.to_csv('results_y_L2.csv', index=False, header=False)
data_y_L3.to_csv('results_y_L3.csv', index=False, header=False)
data_y_L4.to_csv('results_y_L4.csv', index=False, header=False)
data_y_L5.to_csv('results_y_L5.csv', index=False, header=False)
data_y_L15.to_csv('results_y_L15.csv', index=False, header=False)

data_X.to_csv('results_X.csv',index=False, header=False)

