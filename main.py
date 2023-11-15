import pandas as pd
import yfinance as yf
import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class PortfolioOptimizer:
    def __init__(self, start_date, end_date, test_start_date, test_end_date, sharpe_windows):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.test_start_date = pd.to_datetime(test_start_date)
        self.test_end_date = pd.to_datetime(test_end_date)
        self.sharpe_windows = sharpe_windows
        self.prices_df = None  # DataFrame to store stock prices
        self.returns_df = None
        self.excess_returns_df = pd.DataFrame()
        self.logreturns_df = pd.DataFrame()
        self.symbols = []
        self.noa = 0

    def load_data(self, symbols, risk_free_symbol):
        merged_price_df = pd.DataFrame()
        for symbol in symbols:
            self.noa = self.noa + 1
            stock_data = yf.download(symbol, start=self.start_date, end=self.end_date)
            stock_data = stock_data['Adj Close']  # Assuming you want to use Adjusted Close prices
            merged_price_df = merged_price_df.join(stock_data.rename(symbol), how='outer').dropna()
        risk_free_data = yf.download(risk_free_symbol, start=self.start_date, end=self.end_date)
        risk_free_data = risk_free_data['Adj Close'] / 100 / 252
        risk_free_data = risk_free_data.rename("RiskFree")
        merged_df = pd.merge(merged_price_df, risk_free_data, left_index=True, right_index=True, how='outer').dropna()
        self.prices_df = merged_df[symbols]
        self.returns_df = (self.prices_df - self.prices_df.shift(1)) / self.prices_df.shift(1)
        self.returns_df = self.returns_df.dropna()
        for symbol in symbols:
            self.logreturns_df[symbol] = np.log1p(self.returns_df[symbol])
            self.excess_returns_df[symbol] = self.returns_df[symbol] - merged_df["RiskFree"]
        self.excess_returns_df = self.excess_returns_df.dropna()
        self.logreturns_df = self.logreturns_df.dropna()
        self.symbols = symbols

    def predict_returns(self, method="Markowitz"):
        df = self.excess_returns_df
        dates = []
        for date in pd.date_range(start=self.test_start_date, end=self.test_end_date):
            if date in df.index:
                dates.append(date)
        predicted_returns = pd.DataFrame()
        real = df.loc[df.index.isin(dates)]
        if method == "Markowitz":
            window = 5*252
            for date in dates:
                start_idx = df.index.get_loc(date) - window
                end_idx = df.index.get_loc(date)
                window_data = df.iloc[start_idx:end_idx]
                for symbol in self.symbols:
                    predicted_returns.loc[date, symbol] = window_data[symbol].mean()
        if method == "Regression":
            train_start_date = "2010-01-01"
            train_end_date = "2015-12-31"
            df = self.excess_returns_df
            df = df.dropna()
            lag_size = 10
            model = LinearRegression()
            predicted_returns = pd.DataFrame(index=dates)
            for target in df.columns:
                df['Target'] = df[target].shift(-lag_size)
                features = []
                for i in range(lag_size):
                    features.append(f'{target}_lag_{i}')
                    df[f'{target}_lag_{i}'] = df[target].shift(i)
                train_data = df[(df.index >= train_start_date) & (df.index <= train_end_date)]
                test_data = df[(df.index >= self.test_start_date) & (df.index <= self.test_end_date)]
                X_train = train_data.iloc[lag_size:][features]
                y_train = train_data.iloc[lag_size:]['Target']
                model.fit(X_train, y_train)
                X_test = test_data[features]
                predicted_returns[target] = model.predict(X_test)
        return predicted_returns, real

    def optimize_portfolio(self, method="Regression"):
        for window in self.sharpe_windows:
            df = self.excess_returns_df
            optimal_weights_df = pd.DataFrame(index=df.index, columns=df.columns)
            predictions, real = self.predict_returns(method)
            for date in pd.date_range(start=self.test_start_date, end=self.test_end_date):
                if date in df.index:
                    start_idx = df.index.get_loc(date) - window
                    end_idx = df.index.get_loc(date)
                    window_data = df.iloc[start_idx:end_idx]
                    prediction = predictions.loc[date, :]

                    def objective(weights):
                        portfolio_return = np.sum(prediction * weights)
                        portfolio_std = np.sqrt(np.dot(weights, np.dot(window_data.cov() * 252, weights)))
                        SR = portfolio_return / portfolio_std
                        return -SR

                    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Weights sum to 1
                    bounds = tuple((0, 1) for _ in range(self.noa))
                    initial_weights = np.array([1 / self.noa] * self.noa)
                    result = sp.minimize(objective, initial_weights, method='SLSQP', bounds=bounds,
                                         constraints=constraints)
                    optimal_weights_df.loc[date] = result.x
                    print(date)
            optimal_weights_df = optimal_weights_df.dropna()
            portfolio_logreturns = (self.logreturns_df * optimal_weights_df).dropna().sum(axis=1)
            cum_returns = portfolio_logreturns.cumsum()
            portfolio_values = pd.DataFrame()
            portfolio_values["Pf Value" + method] = np.exp(cum_returns.astype(float))
            return optimal_weights_df, portfolio_values

    def plot_results(self, method="Markowitz"):
            optimal_weights_df, portfolio_values = self.optimize_portfolio(method)
            portfolio_values.plot()
            optimal_weights_df.plot(kind="bar", stacked=True, width=1)
            locs, labels = plt.xticks()
            plt.xticks(ticks=locs[0:-1:21], labels=optimal_weights_df.index[0:-1:21].month,
                       rotation="horizontal")
            plt.show()
