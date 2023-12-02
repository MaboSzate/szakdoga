import main as m
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

portfolio = m.PortfolioOptimizer(start_date='2011-01-01', end_date='2018-12-31',
                                 test_start_date='2016-01-01', test_end_date='2018-12-31')
stock_symbols = ['TIP', 'VTV', 'IEF', 'USL', 'GLD']
portfolio.load_data(stock_symbols, '^IRX')


def plot_pf_values():
    aweights, avalues = portfolio.optimize_portfolio(method='Regression')
    bweights, bvalues = portfolio.optimize_portfolio(method='SVR')
    merged = avalues.join(bvalues, on=avalues.index)
    merged.plot()
    plt.show()


def evaluate_predictions(validation_start='2016-01-01', validation_end='2018-12-31', symbol="TIP"):
    portfolio.test_start_date = pd.to_datetime(validation_start)
    portfolio.test_end_date = pd.to_datetime(validation_end)
    methods = ["Markowitz", "RF"]
    predicted, real = portfolio.create_prediction_df(methods=methods, lag_size=20, restrict=False, symbol=symbol)
    metrics = {"MSE": mean_squared_error, "MAE": mean_absolute_error,
               "HR": lambda x, y: np.sum(np.sign(x) == np.sign(y)) / len(x)}
    pred_eval = pd.DataFrame(index=list(metrics))
    for method in predicted.columns:
        for metric in metrics:
            pred_eval.loc[metric, method] = metrics.get(metric)(real, predicted[method])
    pred_eval.loc["MSE", :] *= 1000
    print(symbol, pred_eval)


# plot_pf_values()
# portfolio.plot_results("SVR")
for symbol in portfolio.symbols:
    evaluate_predictions(symbol=symbol)

