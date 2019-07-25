import copy
import math
from typing import Dict, List, Optional
from matplotlib import pyplot
from datetime import date as Date

from experts.obscure_expert import ObscureExpert
from framework.interface_trader import ITrader
from framework.period import Period
from framework.portfolio import Portfolio
from framework.company import Company
from framework.stock_market_data import StockMarketData
from framework.logger import logger
from traders import deep_q_learning_trader
from traders.trusting_trader import TrustingTrader
from traders.trusting_trader2 import TrustingTrader2
from traders.buy_and_hold_trader import BuyAndHoldTrader


class StockExchange:
    """
    This class models the stock exchange where all traders to their trades.
    To prevent cheating, the stock exchange is the golden source of truth for traders portfolios.
    """
    __cash: float
    __trader_portfolios: Dict[ITrader, Dict[Date, Portfolio]]
    __complete_stock_market_data: StockMarketData

    def __init__(self, initial_portfolio_cash: float = 1000.0):
        """
        Constructor
        :param initial_portfolio_cash: The initial cash per portfolio
        """
        self.__cash = initial_portfolio_cash
        self.__trader_portfolios = None
        self.__complete_stock_market_data = None

    def run(self, data: StockMarketData, traders: List[ITrader], offset: int = 0) -> Dict[ITrader, Dict[Date, Portfolio]]:
        """
        Runs the stock exchange over the given stock market data for the given traders.
        :param data: The complete stock market data
        :param traders: A list of all traders
        :param offset: The number of trading days which a will be skipped before (!) trading starts
        :return: The main data structure, which stores one portfolio per trade day, for each traders
        """
        assert data is not None
        assert traders is not None

        # initialize the main data structure: Dictionary over traders, that stores each traders's portfolio per day
        # data structure type is Dict[ITrader, Dict[Date, Portfolio]]
        trade_dates = data.get_trade_days()
        assert trade_dates # must not be empty
        assert 0 <= offset < len(trade_dates) # offset must be feasible
        self.__complete_stock_market_data = data
        self.__trader_portfolios = {trader: {trade_dates[offset]: Portfolio(self.__cash)} for trader in traders}

        # iterate over all trade days minus 1, because we don't trade on the last day
        for tick in range(offset, len(trade_dates) - 1):
            logger.debug(f"Stock Exchange: Current tick '{tick}' means today is '{trade_dates[tick]}'")

            # build stock market data until today
            current_stock_market_data = data.deepcopy_first_n_items(tick + 1)

            # iterate over all traders
            for trader in traders:
                # get the traders's order list by giving him a copy (to prevent cheating) of today's portfolio
                todays_portfolio = self.__trader_portfolios[trader][trade_dates[tick]]
                current_order_list = trader.trade(copy.deepcopy(todays_portfolio), current_stock_market_data)

                # execute order list and save the result as tomorrow's portfolio
                tomorrows_portfolio = copy.deepcopy(todays_portfolio)
                tomorrows_portfolio.update_with_order_list(current_stock_market_data, current_order_list)
                self.__trader_portfolios[trader][trade_dates[tick + 1]] = tomorrows_portfolio

        return self.__trader_portfolios

    def get_final_portfolio_value(self, trader: ITrader) -> float:
        """
        Return the final portfolio value for one traders after (!) the stock exchange ran at least once.
        :param trader: The traders whose final portfolio value will be returned
        :return: The traders's final portfolio value
        """
        assert trader is not None
        assert self.__trader_portfolios is not None
        assert self.__complete_stock_market_data is not None
        final_day = self.__complete_stock_market_data.get_most_recent_trade_day()
        final_portfolio = self.__trader_portfolios[trader][final_day]
        return final_portfolio.get_value(self.__complete_stock_market_data)

    def visualize_last_run(self, barplot: bool = False) -> None:
        """
        Visualize all portfolio values of all traders after (!) the stock exchange ran at least once.
        :return: None
        """
        assert self.__trader_portfolios is not None
        assert self.__complete_stock_market_data is not None
        pyplot.figure(figsize=(8, 5))
        trader_names = []
        for trader in self.__trader_portfolios:
            if isinstance(trader, deep_q_learning_trader.DeepQLearningTrader) and trader.plot_name is not None:
                trader_name = trader.plot_name
            else:
                trader_name = trader.get_name()
            trader_names.append(trader_name)
        if barplot:
            for ti, trader in enumerate(self.__trader_portfolios):
                pyplot.barh(
                    len(trader_names) - 1 - ti,
                    self.get_final_portfolio_value(trader),
                    color=trader.get_color()
                )
            pyplot.title('Comparison of Trader Models')
            pyplot.xlabel('Final Portfolio Value')
            pyplot.yticks(list(range(len(trader_names))), list(reversed(trader_names)))
            pyplot.tight_layout()
        else:
            # pyplot.yscale('log')
            for trader in self.__trader_portfolios:
                portfolios = self.__trader_portfolios[trader]
                keys = portfolios.keys()
                values = [pf.get_value(self.__complete_stock_market_data, date) for date, pf in portfolios.items()]
                pyplot.plot(keys, values, label=trader.get_name(), color=trader.get_color())
            pyplot.title('Comparison of Trader Model Performance')
            pyplot.legend(trader_names)
            pyplot.savefig('traders/stock-exchange-plot.png')
        pyplot.show()


def main_orig():
    """
    Code from the original main routine.
    Visualizes portfolio value over the testing period for different traders.
    """
    # Load stock market data for testing period
    stock_market_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # create new stock exchange with initial portfolio cash for each traders
    stock_exchange = StockExchange(2000.0)

    # create the traders
    bah_trader = BuyAndHoldTrader('black', 'Buy and hold')
    tt_trader_obscure = TrustingTrader(
        ObscureExpert(Company.A), ObscureExpert(Company.B), 'green', 'Trust experts')
    tt_trader2 = TrustingTrader2(
        ObscureExpert(Company.A), ObscureExpert(Company.B), 'limegreen', 'Trust experts for sell only')
    dql_trader = deep_q_learning_trader.DeepQLearningTrader(
        ObscureExpert(Company.A), ObscureExpert(Company.B), True, False, 'red', plot_name='Deep Q-learning trader')

    # run the stock exchange over the testing period, with 100 skipped trading days
    stock_exchange.run(stock_market_data, [bah_trader, tt_trader_obscure, tt_trader2, dql_trader])

    # visualize the results
    stock_exchange.visualize_last_run()


def main_print_model_performance():
    """
    Just prints the final portfolio value and v score (return % per trading day)
    for training and testing data set for different traders.
    """
    stock_market_data_train = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    # stock_market_data_train = stock_market_data_train.deepcopy_first_n_items(
    #     int(stock_market_data_train.get_row_count() / 5))
    stock_market_data_test = StockMarketData([Company.A, Company.B], [Period.TESTING])
    bah_trader = BuyAndHoldTrader(name='buy and hold')
    tt_trader1 = TrustingTrader(
        ObscureExpert(Company.A), ObscureExpert(Company.B), 'green', 'trust experts, prefer A')
    tt_trader2 = TrustingTrader2(
        ObscureExpert(Company.A), ObscureExpert(Company.B), 'limegreen', 'trust experts for sell only')
    dql_trader = deep_q_learning_trader.DeepQLearningTrader(
        ObscureExpert(Company.A), ObscureExpert(Company.B), True, False, 'red')
    all_traders = [bah_trader, tt_trader1, tt_trader2, dql_trader]
    trader_names = []
    for trader in all_traders:
        if isinstance(trader, deep_q_learning_trader.DeepQLearningTrader) and trader.plot_name is not None:
            trader_name = trader.plot_name
        else:
            trader_name = trader.get_name()
        trader_names.append(trader_name)
    max_trader_name_length = max((len(name) for name in trader_names))
    for trader, trader_name in zip(all_traders, trader_names):
        is_first_line = True
        for dataset, dataset_name, starting_cash in [(stock_market_data_train, 'train', 10000.0), (stock_market_data_test, 'test', 2000.0)]:
            stock_exchange = StockExchange(starting_cash)
            stock_exchange.run(dataset, [trader])
            p = stock_exchange.get_final_portfolio_value(trader)
            samples = dataset.get_row_count()
            v = 100.0 * (math.pow(p / starting_cash, 1 / samples) - 1.0)
            if is_first_line:
                header = ('{: <' + str(max_trader_name_length) + '}').format(trader_name)
            else:
                header = ' ' * max_trader_name_length
            header += f' ({dataset_name: <5}): '
            print(f'{header}{v:.5f}% return per trading day'
                  f' (final portfolio value of {p:.1e})')
            is_first_line = False
            if isinstance(trader, BuyAndHoldTrader):
                trader.reset()


if __name__ == "__main__":
    main_orig()  # evaluates all traders over the testing period and visualizes the results
    # main_print_model_performance()
