import typing
from framework.vote import Vote
from dh.IState import IState


input_neurons_per_company = 2
companies = 2
input_neurons = 5


class StateExpertsCashShares(IState):
    """
    Like StateExportsOnly, but additionally feeds the neural network with information
    whether cash is available to buy shares (as a binary information) and
    whether shares of each company are available (as a binary information).
    The idea was to help the agent avoid noneffective actions.
    However, it had no significant impact on the model's performance.
    A possible explanation could be that alternative actions mostly are as noneffective or
    even worse than the originally chosen one, so the additional information does not help.
    """
    def __init__(self, votes: typing.List[Vote], cash: float, shares_available: typing.List[float],
                 portfolio_value: float):
        super(StateExpertsCashShares, self).__init__(portfolio_value)
        assert len(votes) == companies
        assert len(shares_available) == companies
        self.votes = votes
        self.cash = cash
        self.shares_available = shares_available

    def to_input(self) -> typing.List[float]:
        """
        Transforms the state into a list of floats to be fed into the input layer
        of the neural network.
        """

        def vote_to_input_neuron_value(vote: Vote) -> float:
            if vote is Vote.SELL:
                return 0.0
            elif vote is Vote.HOLD:
                return 0.5
            elif vote is Vote.BUY:
                return 1.0
            else:
                raise ValueError

        result = [vote_to_input_neuron_value(vote) for vote in self.votes]
        result.append(1 if self.cash > 0 else 0)
        for share_amount in self.shares_available:
            result.append(1 if share_amount > 0 else 0)

        assert len(result) == input_neurons
        return result

    @staticmethod
    def get_number_of_input_neurons() -> int:
        return input_neurons
