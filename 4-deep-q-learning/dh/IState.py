import abc
import typing


class IState:
    """
    Abstract superclass of classes implementing the concept of state.

    Rather un-pythonic (violates duck-typing principle).
    """
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value

    @abc.abstractmethod
    def to_input(self) -> typing.List[float]:
        raise NotImplementedError

    @abc.abstractstaticmethod
    def get_number_of_input_neurons() -> int:
        raise NotImplementedError
