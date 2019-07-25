import typing
from framework.vote import Vote
from dh.IState import IState


input_neurons_per_company = 1
companies = 2
input_neurons = input_neurons_per_company * companies


class StateExpertsOnly(IState):
    """
    Encapsulates the expert's opinions as the simplest concept of state

    The portfolio value is only needed to make it easier for other code to
    calculate the reward from a state transition.
    """
    def __init__(self, votes: typing.List[Vote], portfolio_value: float):
        super(StateExpertsOnly, self).__init__(portfolio_value)
        assert len(votes) == companies
        self.votes = votes

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

        return [vote_to_input_neuron_value(vote) for vote in self.votes]

    @staticmethod
    def get_number_of_input_neurons() -> int:
        return input_neurons
