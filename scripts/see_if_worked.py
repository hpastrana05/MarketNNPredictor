import numpy as np
from datetime import datetime

def main():

    # Load Time OHLCV
    data = np.loadtxt('./upload/btcusd_1-min_data.csv', delimiter=',', skiprows=1)

    print(data.shape)  # shows rows and columns

    first_day = data[0, 0]
    first_day_time = datetime.fromtimestamp(first_day)
    print("First Date:",first_day_time.strftime('%d/%m/%Y, %H:%M:%S'))

    last_day = data[-1, 0]
    last_day_time = datetime.fromtimestamp(last_day)
    print("Last Date:",last_day_time.strftime('%d/%m/%Y, %H:%M:%S'))

    days = data.shape[0] / 1440
    years = data.shape[0] / (1440*365)
    print("Number days of data (APROX): ", days)
    print("Number years of data (APROX): ", years)


    #Checking 0 values
    close = data[-1]
    close = close[4]
    print(close)

    




if __name__ == "__main__":
    main()

