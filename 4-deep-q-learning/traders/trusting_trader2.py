from typing import List

from framework.company import Company
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.logger import logger
from framework.portfolio import Portfolio
from framework.stock_data import StockData
from framework.stock_market_data import StockMarketData
from framework.order import Order, OrderType
from framework.vote import Vote


class TrustingTrader2(ITrader):
    """
    This trader will sell if the expert says so, and will buy in all other cases.
    When buying both stocks the cash is split evenly among companies (this differs
    from the original trusting trader implementation which prefers company A).
    """

    def __init__(self, expert_a: IExpert, expert_b: IExpert, color: str = 'black', name: str = 'tt_trader'):
        """
        Constructor
        """
        super().__init__(color, name)
        assert expert_a is not None
        assert expert_b is not None
        self.__expert_a = expert_a
        self.__expert_b = expert_b

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate actions to be taken on the "stock market"

        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        order_list = []

        company_list = stock_market_data.get_companies()
        experts = [self.__expert_a, self.__expert_b]
        votes = [
            expert.vote(stock_market_data[comp])
            for comp, expert in zip(company_list, experts)
        ]
        buy_weight_sum = sum((1 for vote in votes if vote is Vote.BUY or vote is Vote.HOLD))
        for comp, vote in zip(company_list, votes):
            buy_weight = 1 / buy_weight_sum if vote is Vote.BUY or vote is Vote.HOLD else None
            self.__follow_expert_vote(comp, stock_market_data[comp], vote, buy_weight, portfolio, order_list)
        return order_list

    def __follow_expert_vote(self, company: Company, stock_data: StockData, vote: Vote, buy_weight: float,
                             portfolio: Portfolio,
                             order_list: List[Order]):
        assert company is not None
        assert stock_data is not None
        assert vote is not None
        assert portfolio is not None
        assert order_list is not None

        if vote is Vote.BUY or vote is Vote.HOLD:
            assert buy_weight is not None and 0 < buy_weight <= 1.0
            stock_price = stock_data.get_last()[-1]
            amount_to_buy = int(buy_weight * portfolio.cash // stock_price)
            logger.debug(f"{self.get_name()}: Got vote to buy {company}: {amount_to_buy} shares a {stock_price}")
            if amount_to_buy > 0:
                order_list.append(Order(OrderType.BUY, company, amount_to_buy))
        elif vote == Vote.SELL:
            # sell as many stocks as possible
            amount_to_sell = portfolio.get_stock(company)
            logger.debug(f"{self.get_name()}: Got vote to sell {company}: {amount_to_sell} shares available")
            if amount_to_sell > 0:
                order_list.append(Order(OrderType.SELL, company, amount_to_sell))
        else:
            # do nothing
            assert vote == Vote.HOLD
            logger.debug(f"{self.get_name()}: Got vote to hold {company}")
