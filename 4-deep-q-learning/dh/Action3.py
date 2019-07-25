import enum
import numpy
import random
import typing


output_neurons_per_action = 3
companies = 2
output_neurons = output_neurons_per_action ** companies


@enum.unique
class Action3(enum.Enum):
    SELL = "sell"
    HOLD = "hold"
    BUY = "buy"

    def is_buy(self):
        if self is Action3.BUY:
            return True
        return False

    def is_sell(self):
        if self is Action3.SELL:
            return True
        return False

    @staticmethod
    def get_random(randgen: typing.Optional[random.Random] = None) -> 'Action3':
        if randgen is None:
            r = random.randrange(0, output_neurons_per_action)
        else:
            r = randgen.randrange(0, output_neurons_per_action)
        return Action3.from_index(r)

    @staticmethod
    def from_index(index: int) -> 'Action3':
        assert 0 <= index < output_neurons_per_action
        return (Action3.SELL, Action3.HOLD, Action3.BUY)[index]

    @staticmethod
    def from_model_prediction(prediction: numpy.ndarray) -> typing.Tuple['Action3', 'Action3']:
        """
        Parses the model output into a list of actions (one per company).

        This variant assumes that each possible action tuple is mapped to its own output neuron,
        i.e. 3*3=9 output neurons.
        """
        assert prediction.shape == (output_neurons, )
        # find index of output neuron with highest value; choose actions based on that index
        (neuron_index, q_value) = max(
            enumerate(prediction),
            key=lambda tup: tup[1]
        )
        return (Action3.from_index(neuron_index // output_neurons_per_action),
                Action3.from_index(neuron_index % output_neurons_per_action))

    @staticmethod
    def from_model_prediction2(prediction: numpy.ndarray) -> typing.Tuple['Action3', 'Action3']:
        """
        Parses the model output into a list of actions (one per company).

        This variant assumes that each possible action per company is mapped to its own output neuron,
        i.e. 3+3=6 output neurons. This representation violates the goal of representing the Q-function
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
            return Action3.from_index(action_index)

        return get_action(0), get_action(1)

    @staticmethod
    def get_action_magnitudes(actions: typing.List['Action3']) -> typing.List[float]:
        """
        Returns a list of floats of length #actions.
        Each entry specifies the relative amount that should be bought or sold.
        Returned values are in the range [0, 1].
        For buy actions the magnitude gives the fraction of cash to invest in the stock.
        For sell actions the magnitude gives the fraction of owned shares to sell.
        For hold actions the returned value is 0.
        """
        def get_buy_weight(action: Action3):
            if action is Action3.BUY:
                return 1
            else:
                return 0

        def get_sell_weight(action: Action3):
            if action is Action3.SELL:
                return 1
            else:
                return 0

        buy_weight_sum = sum((get_buy_weight(a) for a in actions))
        mags = []
        for a in actions:
            buy_weight = get_buy_weight(a)
            if buy_weight > 0:
                mags.append(buy_weight / buy_weight_sum)
            else:
                mags.append(get_sell_weight(a))
        return mags

    @staticmethod
    def get_number_of_output_neurons() -> int:
        return output_neurons

    @staticmethod
    def get_number_of_output_neurons_per_action() -> int:
        return output_neurons_per_action
