import math
import numpy
import random
import typing
from collections import deque
from typing import List
import stock_exchange  # circular dependency!
from experts.obscure_expert import ObscureExpert
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from framework.vote import Vote
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, Nadam, SGD
from framework.order import Company
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger
from dh.StateExpertsOnly import StateExpertsOnly
from dh.StateExpertsCashShares import StateExpertsCashShares
from dh.Action3 import Action3
from dh.Action5 import Action5
from dh.Experience import Experience


State = StateExpertsOnly
Action = Action3


class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = True,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader',
                 other_data_directory: typing.Optional[str] = None,
                 plot_name: typing.Optional[str] = None):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
            other_data_directory: relative directory path from project root to .json and .h5 files of model
            plot_name: name to use in a plot for this trader, overriding parameter "name" (which also determines
                       path where the model gets saved)
        """
        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.experts = [expert_a, expert_b]
        self.train_while_trading = train_while_trading
        self.other_data_directory = other_data_directory
        self.plot_name = plot_name

        # Parameters for neural network
        self.state_size = State.get_number_of_input_neurons()
        self.action_size = Action.get_number_of_output_neurons()
        self.hidden_size = 30

        # Parameters for deep Q-learning
        self.learning_rate = 0.001  # default for Adam: 0.001
        self.epsilon = 0.9995  # determines how quickly epsilon decreases to epsilon_min
        self.random_action_min_probability = 0.01  # minimum probability of choosing a random action
        self.train_each_n_days = 128  # how many trading days pass between each training
        self.batch_size = 128  # how many experience samples from memory to train on each training occasion
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory: typing.Deque[Experience] = deque(maxlen=2000)
        # discount factor: how quickly we expect an action to pay off
        # (0 -> only evaluate immediate effect on portfolio value on next day;
        #  near 1 -> also include development of portfolio value on future days, where each day has a weight of
        #  discount_factor ^ #days_ahead; higher values allow to factor in payoffs at later time while also
        #  making it harder to attribute the effects to the actions from a concrete day)
        self.discount_factor = 0
        assert 0 <= self.discount_factor < 1.0
        # min_horizon: number of more recent experiences required until discount_factor ^ #days_ahead drops to 1%
        if 0 < self.discount_factor < 1:
            self.min_horizon = min(100, math.ceil(math.log(0.01, self.discount_factor)))
        else:
            self.min_horizon = 1
        self.q_value_cap = 1.0  # limit Q-value to this magnitude; 0.0 to deactivate
        self.is_evolved_model = False  # set to True when loading model from file
        self.days_passed = 0
        self.training_occasions = 0

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            data_dir = self.RELATIVE_DATA_DIRECTORY if self.other_data_directory is None else self.other_data_directory
            self.model = load_keras_sequential(data_dir, 'dql_trader')
            # logger.info(f"DQL Trader: Loaded trained model")
            self.is_evolved_model = True
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            # original code:
            # self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu'))
            # self.model.add(Dense(self.hidden_size, activation='relu'))
            # self.model.add(Dense(self.action_size, activation='linear'))
            # modified code:
            # relu -> elu: avoid "dead nodes" problem of relu
            # lecun_normal: initialization from gaussian accounting for number of nodes as well
            # initialize bias with zeros
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='elu', kernel_initializer='lecun_normal', bias_initializer='zeros'))
            self.model.add(Dense(self.hidden_size, activation='elu', kernel_initializer='lecun_normal', bias_initializer='zeros'))
            self.model.add(Dense(self.action_size, activation='linear', kernel_initializer='lecun_normal', bias_initializer='zeros'))
            # logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        # use one of the following solvers:
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        # self.model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
        # self.model.compile(loss='mse', optimizer=Nadam(lr=0.001))

    def log_info_every_nth_day(self, msg: str, n: int):
        if self.days_passed % n == 0:
            logger.info(msg)

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        data_dir = self.RELATIVE_DATA_DIRECTORY if self.other_data_directory is None else self.other_data_directory
        save_keras_sequential(self.model, data_dir, 'dql_trader')
        # logger.info(f"DQL Trader: Saved trained model")

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate actions to be taken on the "stock market"
    
        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]

        # Compute the current state
        expert_votes = [
            self.experts[i].vote(stock_market_data[company])
            for i, company in enumerate(stock_market_data.get_companies())
        ]
        shares_owned = [
            portfolio.get_stock(company)
            for company in stock_market_data.get_companies()
        ]
        if State is StateExpertsOnly:
            state = StateExpertsOnly(expert_votes, portfolio.get_value(stock_market_data))
        elif State is StateExpertsCashShares:
            state = StateExpertsCashShares(expert_votes, portfolio.cash, shares_owned, portfolio.get_value(stock_market_data))
        else:
            raise RuntimeError

        if self.train_while_trading:
            # store state as experience in memory
            if len(self.memory) > 0:
                self.memory[-1].state2 = state
            experience = Experience(
                state1=state
            )
            self.memory.append(experience)

            # train
            if len(self.memory) >= self.min_size_of_memory_before_training:
                if self.days_passed % self.train_each_n_days == 0:
                    self.train()

            # determine probability for random actions
            if not self.is_evolved_model:
                # first training episode
                random_action_probability = (
                        (self.epsilon ** self.days_passed) * (1.0 - self.random_action_min_probability) +
                        self.random_action_min_probability
                )
            else:
                # subsequent training episode
                random_action_probability = self.random_action_min_probability

            if self.training_occasions == 0 or random.random() < random_action_probability:
                actions = [Action.get_random(), Action.get_random()]
            else:
                # choose actions by querying network
                x = state.to_input()
                y = self.model.predict(numpy.array([x]))
                assert y.shape == (1, self.action_size)
                actions = Action.from_model_prediction(y[0])

            experience.actions = actions
        else:
            # not training -> always choose actions by querying network
            actions = Action.from_model_prediction(self.model.predict(numpy.array([state.to_input()]))[0])

        # translate actions into orders
        orders: typing.List[Order] = []
        companies_with_actions_and_magnitudes = list(zip(list(Company), actions, Action.get_action_magnitudes(actions)))
        for comp, action, mag in companies_with_actions_and_magnitudes:
            if action.is_buy():
                cash_limit = portfolio.cash * mag
                date, stock_price = stock_market_data[comp].get_last()
                shares_amount = cash_limit / stock_price
                if shares_amount > 0:
                    orders.append(Order(OrderType.BUY, comp, shares_amount))
            elif action.is_sell():
                shares_amount = portfolio.get_stock(comp) * mag
                if shares_amount > 0:
                    orders.append(Order(OrderType.SELL, comp, shares_amount))

        self.days_passed += 1
        return orders

    def train(self):
        """
        Train model based on experiences stored in memory.
        To speed up training a batch of experiences is trained at once (instance attribute batch_size).
        Conversely, training can happen at intervals up to batch_size days (instance attribute .
        """
        # determine cumulative rewards for actions in memory
        # loop backwards through the memory, ignoring the latest entry
        for i in range(2, len(self.memory) + 1):
            e: Experience = self.memory[-i]
            next_e: Experience = self.memory[-i + 1]  # its reward has been set in the previous loop iteration
            if i == 2:
                e.reward = e.get_portfolio_rel_immediate_change() - 1
            else:
                e.reward = e.get_portfolio_rel_immediate_change() - 1 + self.discount_factor * next_e.reward
        # modify reward as part of the computation of the Q-value:
        # 1. multiply by 100 to make it easier for the network to learn the values
        #    (brings absolute values mostly into a range of 0.01 .. 10.0)
        # 2. normalize for time horizon
        #    (assuming a high discount factor there is a large difference in cumulative reward between
        #     very recent and rather old experiences, simply because accumulating (discounted) reward
        #     over more recent experiences stops when the most recent experience has been hit)
        discounted_time_factor = 1.0
        for i in range(2, len(self.memory) + 1):
            e: Experience = self.memory[-i]
            e.reward *= 100.0 / discounted_time_factor
            discounted_time_factor += self.discount_factor ** (i - 1)

        # A part of the memory is too recent, i.e. the rewards for these states have only a short time horizon.
        # We want to exclude these recent states from learning because they do not meet the requirements
        # implied by the discount factor.
        memory_range_for_training = range(len(self.memory) - self.min_horizon)

        # assemble input (x) and output (y) arrays for training
        batch_indices = random.sample(memory_range_for_training, self.batch_size)
        x = numpy.array([
            State.to_input(self.memory[i].state1) for i in batch_indices
        ])
        assert x.shape == (self.batch_size, self.state_size)
        # use the output of the network as the basis for training
        y = self.model.predict(x)
        assert y.shape == (self.batch_size, self.action_size)
        action_to_index = {a: i for (i, a) in enumerate(Action)}
        # modify those parts of the output that correspond to the actions taken in the
        # experiences selected from memory
        for bi in range(self.batch_size):
            e: Experience = self.memory[batch_indices[bi]]
            q_value_to_train = e.reward

            # cap Q-value to lower model performance fluctuations during training
            if self.q_value_cap > 0.0:
                if q_value_to_train > self.q_value_cap:
                    q_value_to_train = self.q_value_cap
                if q_value_to_train < -self.q_value_cap:
                    q_value_to_train = -self.q_value_cap

            # 1) if encode each possible action tuple as a separate output neuron
            #    e.g. Action5 with output layer as 5^(#companies) neurons
            node_index = (action_to_index[e.actions[0]] * Action.get_number_of_output_neurons_per_action() +
                          action_to_index[e.actions[1]])
            y[bi][node_index] = q_value_to_train

            # 2) if encode each possible action per company as separate output neurons
            #    e.g. Action5 with output layer as 5*(#companies) neurons
            #    (simplified representation to reduce #output neurons)
            # node_index = action_to_index[e.actions[0]]
            # y[bi][node_index] = q_value_to_train
            # node_index = action_to_index[e.actions[1]] + Action.get_number_of_output_neurons_per_action()
            # y[bi][node_index] = q_value_to_train

            # debug info: show some untrained vs. trained Q-values
            # if bi == 0:
            #     self.log_info_every_nth_day(f'{y[bi][node_index]:.6f} vs. {q_value_to_train:.6f}', 50)
        self.model.train_on_batch(x, y)
        self.training_occasions += 1

    def print_policy(self):
        """
        Print the policy by enumerating all possible inputs and their corresponding output
        (actions determined by the highest Q-value).
        """
        if State is not StateExpertsOnly:
            return
        for eo1 in list(Vote):
            for eo2 in list(Vote):
                state = StateExpertsOnly([eo1, eo2], 0)
                x = state.to_input()
                y = self.model.predict(numpy.array([x]))[0]
                actions = Action.from_model_prediction(y)
                x_str = f'[{eo1.value: <4}, {eo2.value: <4}]'
                if Action is Action3:
                    action_str = f'[{actions[0].value: <4}, {actions[1].value: <4}]'
                elif Action is Action5:
                    action_str = f'[{actions[0].value: <11}, {actions[1].value: <11}]'
                else:
                    raise NotImplementedError
                y_str = ' '.join((f'{value:+.4f}' for value in y))
                logger.info(f'{x_str} -> {action_str} ({y_str})')


# Executing this module retrains the deep q-learning trader from scratch

# trains TRAINING_RUNS different models and saves the one with the median v score (return_pct_per_day)
TRAINING_RUNS = 5

# training is finished after data from the training set has been replayed EPISODES times
EPISODES = 5

if __name__ == "__main__":
    # Create the training data and testing data
    data = dict(train=StockMarketData([Company.A, Company.B], [Period.TRAINING]),
                test=StockMarketData([Company.A, Company.B], [Period.TESTING]))
    print(f"training data: {data['train'].get_row_count()} samples")
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    # data['train'] = data['train'].deepcopy_first_n_items(int(data['train'].get_row_count() / 5))
    print(f"training data (cropped): {data['train'].get_row_count()} samples")
    print(f"testing data: {data['test'].get_row_count()} samples")

    # Save final portfolio values and return % per day
    # per training run and episode
    final_portfolio_values = dict(train=[], test=[])
    return_pct_per_day = dict(train=[], test=[])
    traders = dict(train=[], test=[])

    def get_last_portfolio_value(phase: str, run: int) -> float:
        index = min(run * EPISODES + EPISODES - 1, len(final_portfolio_values[phase]) - 1)
        return final_portfolio_values[phase][index]

    def get_last_v_score(phase: str, run: int) -> float:
        index = min(run * EPISODES + EPISODES - 1, len(return_pct_per_day[phase]) - 1)
        return return_pct_per_day[phase][index]

    for run in range(TRAINING_RUNS):
        # Create the stock exchange and one traders to train the net
        starting_cash = dict(train=10000.0, test=2000.0)
        stock_exchanges = dict(train=stock_exchange.StockExchange(starting_cash["train"]),
                               test=stock_exchange.StockExchange(starting_cash["test"]))
        traders['train'].append(DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True))

        for episode in range(EPISODES):
            # logger.info(f"DQL Trader: Starting training episode {episode}")

            for phase in ['train', 'test']:
                if phase == 'test':
                    testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
                    if episode == 0:
                        traders[phase].append(testing_trader)
                    else:
                        traders[phase][-1] = testing_trader  # replace testing trader from previous episode
                trader = traders[phase][-1]
                stock_exchanges[phase].run(data[phase], [trader])
                if phase == 'train':
                    trader.save_trained_model()  # required to be able to create testing trader in next iteration
                p = stock_exchanges[phase].get_final_portfolio_value(trader)
                v = 100.0 * (math.pow(p / starting_cash[phase], 1 / data[phase].get_row_count()) - 1.0)
                final_portfolio_values[phase].append(p)
                return_pct_per_day[phase].append(v)

            logger.info(f"DQL Trader: Finished training episode {episode}, "
                        f"final portfolio value training {get_last_portfolio_value('train', run):.1e} vs. "
                        f"final portfolio value test {get_last_portfolio_value('test', run):.1e}")
            logger.info(f"\treturn % per day training {get_last_v_score('train', run):.6f} vs. "
                        f"return % per day test {get_last_v_score('test', run):.6f}")
            # traders['train'][-1].print_policy()

        logger.info('-' * 80)

    # sort traders by test v score
    traders_sorted_by_score = sorted(enumerate(traders['test']), key=lambda tup: get_last_v_score('test', tup[0]))
    # choose the one in the middle
    run, trader = traders_sorted_by_score[(TRAINING_RUNS - 1) // 2]
    # overwrite model from other training runs
    trader.save_trained_model()
    print(f"choosing model with median v score from run {run + 1}")
    print('output for csv data:')
    print('trader;dataset;returnpctperday;finalportfoliovalue')
    print('-' * 80)
    print(f";train;{get_last_v_score('train', run):.6f};{get_last_portfolio_value('train', run):.1e}")
    print(f";test;{get_last_v_score('test', run):.6f};{get_last_portfolio_value('test', run):.1e}")
    print('-' * 80)

    # plotting v scores across training episodes
    from matplotlib import pyplot as plt
    plt.figure()
    data_start_index = run * EPISODES
    data_stop_index = data_start_index + EPISODES
    plt.plot(return_pct_per_day['train'][data_start_index:data_stop_index], label='training', color="black")
    plt.plot(return_pct_per_day['test'][data_start_index:data_stop_index], label='testing', color="green")
    plt.title('v score training vs. testing')
    plt.ylabel('v score (avg. return % per day)')
    plt.xlabel('episode')
    plt.legend(['training', 'testing'])
    plt.show()
