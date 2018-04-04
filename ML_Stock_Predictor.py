"""
ML_StockAnalysis.py
This file will analyze stock prices from the previous three months until now,
and use them to predict future prices for a certain ticker symbol.

Author: Vincent Chao
"""

import datetime  # Date/Time manipulation
import numpy as np  # Perform calculations on data
import matplotlib.pyplot as plt  # Graphically plot data points and models
import fix_yahoo_finance as yf  # Retrieve data from Yahoo Finance
from sklearn.svm import SVR  # Build a predictive model
from sklearn.model_selection import train_test_split  # Split the dataset into training and testing sets


# Set current reference time
now = datetime.datetime.now()
today = datetime.date(now.year, now.month, now.day)


def get_historical(symbol):
    """
    Retrieve historical data file from Yahoo Finance
    :param symbol: Ticker symbol
    :return: DataFrame containing historical data
    """
    year = now.year
    if now.month < 4:
        month = now.month + 9
        year = year - 1
    else:
        month = now.month - 3
    return yf.download(symbol,
                       start="{0}-{1}-{2}".format(year, month, now.day),
                       end=today.strftime('%Y-%m-%d'))


def get_data(data):
    """
    Pull the dates and closing prices from a DataFrame object
    :param data: DataFrame object retrieved from get_historical_data(...)
    :return: 1-dim array of dates and corresponding 1-dim array of prices
    """
    dates = []
    prices = []

    for index, row in data.iterrows():
        date = index.to_pydatetime()
        dates.append((date - now).days)
        prices.append(row['Close'])
    dates, dates_test, prices, prices_test = train_test_split(dates, prices, test_size=0.2, shuffle=False)
    return dates, dates_test, prices, prices_test


def predict_price(dates, dates_test, prices, prices_test, guess, symbol):
    """
    Use Support Vector Regression algorithms to model the price based on time
    :param dates: 1-dim dates array of training data
    :param dates_test: 1-dim dates array of testing data
    :param prices: 1-dim prices array of training data
    :param prices_test: 1-dim prices array of testing data
    :param guess: Float guess of stock price
    :param symbol: Ticker symbol of stock
    :return: Float predictions of stock price based on ML algorithms
    """
    dates = np.reshape(dates, (len(dates), 1))

    # Set parameters for each of the models used
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    # Fit the models based on the training data
    svr_lin.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    # Plot training data as a scatter plot
    plt.scatter(dates, prices, color='black', label='Data')

    # Plot each of the models
    plt.plot(dates, svr_lin.predict(dates), color='blue', label='Linear model')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')

    # Score the models
    dates_test_array = np.asarray(dates_test).reshape(-1, 1)
    prices_test_array = np.asarray(prices_test).reshape(-1, 1)
    score_lin = svr_lin.score(dates_test_array, prices_test_array)
    score_rbf = svr_rbf.score(dates_test_array, prices_test_array)

    # Other Plot Parameters
    plt.xlabel('Date Delta from Today')
    plt.ylabel('Prices')
    plt.title('Support Vector Regression: ' + symbol)
    plt.legend()
    plt.show(block=False)  # Nonblocking, requires extra show() command at end to retain plot window
    return svr_lin.predict(1)[0], score_lin, svr_rbf.predict(1)[0], score_rbf


def main():
    """
    Main Function
    :return: none
    """
    # Initial guess for stock price
    guess_price = 30

    # Ask user what symbol to analyze
    ticker_symbol = input("Enter a symbol: ")

    while True:
        # try:
        data = get_historical(ticker_symbol)
        dates, dates_test, prices, prices_test = get_data(data)
        predicted_price_lin, score_lin, predicted_price_rbf, score_rbf = predict_price(dates, dates_test,
                                                                                       prices, prices_test,
                                                                                       guess_price, ticker_symbol)

        print(
            'Prediction for:', ticker_symbol, '\n\n',
            'Model: SVR/Linear\n',
            'Most Recent Price: $%g\n' % round(prices[-1], 2),
            'Predicted Price:   $%g\n' % round(predicted_price_lin, 2),
            'Change:            $%g\n' % round(predicted_price_lin - prices[-1], 2),
            'Score:              %g\n\n' % score_lin,
            'Model: SVR/Radial Basis Function\n',
            'Most Recent Price: $%g\n' % round(prices[-1], 2),
            'Predicted Price:   $%g\n' % round(predicted_price_rbf, 2),
            'Change:            $%g\n' % round(predicted_price_rbf - prices[-1], 2),
            'Score:              %g\n\n' % score_rbf,
        )
        plt.show()  # Keeps plot window open
        break
        # except ValueError:
        #     ticker_symbol = input("Please enter a valid symbol: ")


if __name__ == "__main__":
    main()