import pandas as pd
# print(pd.__version__)

CSV_FILE_LINK = "https://www.multipasko.pl/wyniki-csv.php?f=minilotto-sortowane"
data = pd.read_csv(CSV_FILE_LINK, delimiter=';')
# print(data.tail(5))
data = data.tail(100)
data_L1 = data.drop(columns=['Numer', 'Miesiac', 'Rok', 'L2', 'L3', 'L4', 'L5'])
data_L1 = data_L1.rename(columns={'Dzien': 'X', 'L1': 'y'})
# data_L2 = data.drop(columns=['Numer', 'Miesiac', 'Rok', 'L1', 'L3', 'L4', 'L5'])
# data_L3 = data.drop(columns=['Numer', 'Miesiac', 'Rok', 'L1', 'L2', 'L4', 'L5'])
# data_L4 = data.drop(columns=['Numer', 'Miesiac', 'Rok', 'L1', 'L2', 'L3', 'L5'])
# data_L5 = data.drop(columns=['Numer', 'Miesiac', 'Rok', 'L1', 'L2', 'L3', 'L4'])
# print(f"{data_L1.tail(5)}")
data_L1.to_csv('results_L1.csv', index=False, header=False)
# data_L2.to_csv('results_L2.csv', index=False, header=False)
# data_L3.to_csv('results_L3.csv', index=False, header=False)
# data_L4.to_csv('results_L4.csv', index=False, header=False)
# data_L5.to_csv('results_L5.csv', index=False, header=False)