import main as m
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

portfolio = m.PortfolioOptimizer(start_date='2007-01-01', end_date='2023-03-01',
                                 test_start_date='2016-01-01', test_end_date='2022-12-31',
                                 sharpe_windows=[5*252])
stock_symbols = ['TIP', 'VTV', 'IEF', 'USL', 'GLD']

# Load stock price data and risk-free returns data
portfolio.load_data(stock_symbols, '^IRX')
# mpred, real = portfolio.predict_returns(5*252, 'Markowitz')
# rpred, real2 = portfolio.predict_returns(5*252, 'Regression')
# print(mean_squared_error(real2, mpred) * 100, mean_squared_error(real2, rpred) * 100)

mweights, mvalues = portfolio.optimize_portfolio(method='Markowitz')
weights, rvalues = portfolio.optimize_portfolio(method='Regression')
merged = mvalues.join(rvalues, on=mvalues.index)
merged.plot()
plt.show()
# portfolio.plot_results('Regression')


