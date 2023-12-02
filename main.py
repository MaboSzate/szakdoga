import pandas as pd
import yfinance as yf
import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


class PortfolioOptimizer:
    def __init__(self, start_date, end_date, test_start_date, test_end_date):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.test_start_date = pd.to_datetime(test_start_date)
        self.test_end_date = pd.to_datetime(test_end_date)
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

    def predict_returns(self, method="Markowitz", lag_size=10, restrict=False, symbol="TIP", window=252*5):
        df = self.excess_returns_df.dropna()
        test_dates = []
        for date in pd.date_range(start=self.test_start_date, end=self.test_end_date):
            if date in df.index:
                test_dates.append(date)
        predicted_returns = pd.DataFrame(index=test_dates)
        real = df.loc[df.index.isin(test_dates)]
        if method == "Markowitz":
            for date in test_dates:
                start_idx = df.index.get_loc(date) - window
                end_idx = df.index.get_loc(date)
                window_data = df.iloc[start_idx:end_idx]
                predicted_returns.loc[date, symbol] = window_data[symbol].mean()
        else:
            data = pd.DataFrame(index=self.excess_returns_df.index)
            train_data = pd.DataFrame()
            test_data = pd.DataFrame()
            data['Target'] = df[symbol]
            features = []
            for i in range(lag_size):
                features.append(f'{symbol}_lag_{i + 1}')
                data[f'{symbol}_lag_{i + 1}'] = df[symbol].shift(i + 1)
            train_start_date = pd.to_datetime(self.start_date)
            test_start_date = pd.to_datetime(self.test_start_date)
            test_end_date = pd.to_datetime(self.test_end_date)
            while test_start_date <= test_end_date:
                yield data[(data.index >= train_start_date) & (data.index < test_start_date)].dropna(), data[
                    (data.index >= test_start_date) & (data.index < test_start_date + pd.DateOffset(years=1))].dropna()
                train_start_date += pd.DateOffset(years=1)
                test_start_date += pd.DateOffset(years=1)
            X_train = train_data[features]
            y_train = train_data['Target']
            X_test = test_data[features]
            if method == "Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                predicted_returns[symbol] = model.predict(X_test)
            if method == "SVR":
                scaler = StandardScaler()
                target_scaler = StandardScaler()
                y_train = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
                if restrict:
                    for feature in features:
                        d = X_train[feature]
                        dm = d.median()
                        dmm = d.sub(dm).abs().median()
                        X_train.loc[X_train[feature] > dm + 5 * dmm, feature] = dm + 5 * dmm
                        X_train.loc[X_train[feature] < dm - 5 * dmm, feature] = dm - 5 * dmm
                        d = X_test[feature]
                        dm = d.median()
                        dmm = d.sub(dm).abs().median()
                        X_test.loc[X_test[feature] > dm + 5 * dmm, feature] = dm + 5 * dmm
                        X_test.loc[X_test[feature] < dm - 5 * dmm, feature] = dm - 5 * dmm
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                # param_grid = {'C': [2**x for x in range(6)], 'gamma': [2**x for x in range(-5, 1)]}
                # grid_search = GridSearchCV(estimator=SVR(kernel='rbf'), param_grid=param_grid, cv=3,
                #                           scoring='neg_mean_squared_error')
                # grid_search.fit(X_train, y_train)
                # opt_C = grid_search.best_params_.get("C")
                # opt_g = grid_search.best_params_.get("gamma")
                # print(symbol, opt_C, opt_g)
                opt_C, opt_g = 1, 1
                if symbol == "VTV":
                    opt_g = 0.03125
                model = SVR(C=opt_C, kernel='rbf', gamma=opt_g)
                model.fit(X_train, y_train)
                pred = model.predict(X_test).reshape(1, -1)
                pred = target_scaler.inverse_transform(pred)
                predicted_returns[symbol] = pred.reshape(-1, 1)
            if method == "RF":
                # param_grid = {'max_depth': [5 * x for x in range(3, 5)], 'min_samples_split': [5 * x for x in range(1, 3)],
                #              'min_samples_leaf': [5 * x for x in range(1, 3)]}
                # random_search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=42),
                #                                  param_distributions=param_grid, cv=3, scoring='neg_mean_squared_error',
                #                                   n_iter=4, random_state=42, n_jobs=-1)
                # random_search.fit(X_train, y_train)
                # opt_depth = random_search.best_params_.get("max_depth")
                # opt_split = random_search.best_params_.get("min_samples_split")
                # opt_leaf = random_search.best_params_.get("min_samples_leaf")
                #opt_features = random_search.best_params_.get("max_features"
                if symbol in ['TIP', 'IEF', 'GLD']:
                    opt_depth = 20
                else:
                    opt_depth = 15
                if symbol in ['TIP', 'GLD']:
                    opt_leaf = 10
                else:
                    opt_leaf = 5
                model = RandomForestRegressor(n_estimators=500, max_depth=opt_depth, min_samples_split=10,
                                              min_samples_leaf=opt_leaf)
                model.fit(X_train, y_train)
                X_test = test_data[features]
                predicted_returns[symbol] = model.predict(X_test)
            if method == "rnd":
                for date in test_dates:
                    for symbol in self.symbols:
                        predicted_returns.loc[date, symbol] = np.random.normal(0, 0.1 / 250)
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

    def create_prediction_df(self, methods, lag_size=10, restrict=False, symbol="TIP"):
        predictions = pd.DataFrame()
        for method in methods:
            preds, real = self.predict_returns(method, lag_size, restrict, symbol)
            preds, real = preds[symbol], real[symbol]
            predictions[method] = preds
        return predictions, real

    def predict_rolled_returns(self, method, lag_size=10, restrict=False, symbol="TIP"):

        pass
