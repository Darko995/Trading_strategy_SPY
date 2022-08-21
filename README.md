# Trading_strategy_SPY
The trading strategy was developed in a python programming language.

Data collection

Data of the Standard & Poor's 500 ETF (Ticker Symbol: SPY) from Jan 1st 2000 to Dec 31st 2010 was downloaded from Yahoo Finance site. Data consists of 2676 rows and 6 columns (High, Low, Open, Close, Volume, Adj Close).
Before constructing the model we will look at what the data look like:


On the y axis is stock price at the end of the day. We can notice that the value of these stocks has fallen in this time period, so if we invested money in the beginning we would eventually lose a certain amount of money.
The idea is to form and train a classification model on the basis of historical data, on the basis of which we will predict the movement of stocks in the future with as much accuracy as possible and on the basis of these predictions buy or sell stocks.

