import enum
import numpy
import random
import typing


output_neurons_per_action = 5
companies = 2
output_neurons = output_neurons_per_action ** companies
strong_factor = 4


@enum.unique
class Action5(enum.Enum):
    STRONG_SELL = "strong sell"
    WEAK_SELL = "weak sell"
    HOLD = "hold"
    WEAK_BUY = "weak buy"
    STRONG_BUY = "strong buy"

    def is_buy(self):
        if self is Action5.STRONG_BUY or self is Action5.WEAK_BUY:
            return True
        return False

    def is_sell(self):
        if self is Action5.STRONG_SELL or self is Action5.WEAK_SELL:
            return True
        return False

    @staticmethod
    def get_random(randgen: typing.Optional[random.Random] = None) -> 'Action5':
        if randgen is None:
            r = random.randrange(0, output_neurons_per_action)
        else:
            r = randgen.randrange(0, output_neurons_per_action)
        return Action5.from_index(r)

    @staticmethod
    def from_index(index: int) -> 'Action5':
        assert 0 <= index < output_neurons_per_action
        return (Action5.STRONG_SELL, Action5.WEAK_SELL, Action5.HOLD, Action5.WEAK_BUY, Action5.STRONG_BUY)[index]

    @staticmethod
    def from_model_prediction(prediction: numpy.ndarray) -> typing.Tuple['Action5', 'Action5']:
        """
        Parses the model output into a list of actions (one per company).

        This variant assumes that each possible action tuple is mapped to its own output neuron,
        i.e. 5*5=25 output neurons.
        """
        assert prediction.shape == (output_neurons, )
        # find index of output neuron with highest value; choose actions based on that index
        (neuron_index, q_value) = max(
            enumerate(prediction),
            key=lambda tup: tup[1]
        )
        return (Action5.from_index(neuron_index // output_neurons_per_action),
                Action5.from_index(neuron_index % output_neurons_per_action))

    @staticmethod
    def from_model_prediction2(prediction: numpy.ndarray) -> typing.Tuple['Action5', 'Action5']:
        """
        Parses the model output into a list of actions (one per company).

        This variant assumes that each possible action per company is mapped to its own output neuron,
        i.e. 5+5=10 output neurons. This representation violates the goal of representing the Q-function
        as a neural network, but the model's performance is comparable or (in the case of 5 actions)
        much better than with the strict tuple representation.
        """
        assert prediction.shape == (output_neurons, )

        def get_action(company_index):
            shift_by = output_neurons_per_action * company_index
            # find index of output neuron with highest value; choose actions based on that index
            (action_index, q_value) = max(
                enumerate(prediction[shift_by: shift_by + output_neurons_per_action]),
                key=lambda tup: tup[1]
            )
            return Action5.from_index(action_index)

        return get_action(0), get_action(1)

    @staticmethod
    def get_action_magnitudes(actions: typing.List['Action5']) -> typing.List[float]:
        """
        Returns a list of floats of length #actions.
        Each entry specifies the relative amount that should be bought or sold.
        Returned values are in the range [0, 1].
        For buy actions the magnitude gives the fraction of cash to invest in the stock.
        For sell actions the magnitude gives the fraction of owned shares to sell.
        For hold actions the returned value is 0.
        """
        def get_buy_weight(action: Action5):
            if action is Action5.STRONG_BUY:
                return strong_factor
            elif action is Action5.WEAK_BUY:
                return 1
            else:
                return 0

        def get_sell_weight(action: Action5):
            if action is Action5.STRONG_SELL:
                return strong_factor
            elif action is Action5.WEAK_SELL:
                return 1
            else:
                return 0

        buy_weight_sum = max(strong_factor, sum((get_buy_weight(a) for a in actions)))
        mags = []
        for a in actions:
            buy_weight = get_buy_weight(a)
            if buy_weight > 0:
                mags.append(buy_weight / buy_weight_sum)
            else:
                mags.append(get_sell_weight(a) / strong_factor)
        return mags

    @staticmethod
    def get_number_of_output_neurons() -> int:
        return output_neurons

    @staticmethod
    def get_number_of_output_neurons_per_action() -> int:
        return output_neurons_per_action
