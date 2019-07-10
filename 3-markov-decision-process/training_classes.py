from data_exploration_classes import Data, GeneratedData
from problem_description_and_state import ProblemDescription, State, ColorCountState, InventoryLevelState

import numpy
import scipy.sparse
import typing
from typing import List, Optional


class SparseMatrix:
    """
    Used by TransitionProbabilityLearner to count state transitions.

    Implemented as dictionaries inside dictionaries. Starts counting from 0.
    """
    def __init__(self):
        self.default = 0
        self.rows = dict()

    def get(self, row, col):
        """
        Returns the number of observed transitions from a certain state
        to another certain state.

        Parameters
        ----------
        row
            State index from which the transition occurs.
        col
            State index to which the transition occurs.
        """
        if row not in self.rows:
            return self.default
        if col not in self.rows[row]:
            return self.default
        return self.rows[row][col]

    def inc(self, row: int, col: int):
        """
        Counting a transition by incrementing a value by 1.

        Parameters
        ----------
        row
            State index from which the transition occurs.
        col
            State index to which the transition occurs.
        """
        if row not in self.rows:
            self.rows[row] = dict()
        if col not in self.rows[row]:
            self.rows[row][col] = self.default
        self.rows[row][col] += 1

    def get_row_sum(self, row):
        """
        Returns the number of observed transitions from a certain state.

        Parameters
        ----------
        row
            State index from which the transitions occur.
        """
        if row not in self.rows:
            return self.default
        return sum((self.rows[row][col] for col in self.rows[row]))

    def get_number_of_entries(self):
        """
        Returns the number of explicit (i.e. non-default) values in the matrix.
        """
        return sum((len(row_entries) for row_entries in self.rows.values()))


class TransitionProbabilityLearner:
    """
    Counts state transitions observed in training data and derives transition
    probabilities from these results, which is used by
    TransitionProbabilityMatrix to populate its entries.

    States are simplified to ignore the order in which the items are stored
    in the inventory before counting a transition. This functionality is
    provided by the ColorCountState class. A further simplification which
    ignores color information and only accounts for the number of items in
    the inventory and the type of verb is provided by the InventoryLevelState
    class. The counts for the latter are used as a fallback if there are not
    enough observed transitions for the former. Finally it can fall back to
    only using the observed color distribution (e.g. for transitions that
    did not occur in the training data).

    Parameters
    ----------
    pd
        Describes the problem (e.g. inventory shape, verbs, colors).
    min_support
        How many transitions must at least have been observed in order to
        use the results from a certain matrix used for counting.
        The value is multiplied by the number of states that can
        be transitioned to (e.g. multiplied by a maximum of 6 if the problem
        consists of 2 verbs and 3 colors).
        get_transition_probabilities() will switch to a less granular matrix
        if neccessary or even fall back to assuming uniform verb distribution
        and only using the learned color distribution.
    """
    def __init__(self, pd: ProblemDescription, data: typing.Union[Data, GeneratedData], min_support: int = 5):
        self.pd = pd
        self.min_support = min_support
        self.transition_count = 0
        self.color_count_matrix = SparseMatrix()
        self.inventory_level_matrix = SparseMatrix()
        self.color_count = [0] * pd.number_of_colors
        self.data = data
        self.learn_from_data()

    def learn_from_data(self):
        """
        Populate the matrices for counting with transitions observed in the
        given data.
        """
        pd = self.pd
        ccs = None
        ils = None
        for (verb, color) in self.data.requests:
            self.color_count[color] += 1
            if ccs is None:
                ccs = ColorCountState([0] * pd.number_of_colors, verb, color)
                ils = InventoryLevelState(0, verb)
                continue
            next_ccs = ccs.apply_action(pd, 0, next_verb=verb, next_color=color)
            next_ils = ils.apply_action(pd, 0, next_verb=verb)
            self.color_count_matrix.inc(ccs.get_index(pd), next_ccs.get_index(pd))
            self.inventory_level_matrix.inc(ils.get_index(pd), next_ils.get_index(pd))
            self.transition_count += 1
            ccs = next_ccs
            ils = next_ils

    def get_transition_probabilities(self, from_state: State, to_states: typing.List[State]
                                     ) -> typing.List[typing.Tuple[State, float]]:
        """
        Used by TransitionProbabilityMatrix to get the transition probabilities
        from a certain state to a number of successor states.


        If there is not enough data to estimate the probability (controlled
        through min_support) it will fall back

        Returns
        -------
        A list of tuples. Each tuple specifies a successor state and the
        probability for the transition. The probabilities are normalized
        (i.e. they sum up to 1.0).
        """
        pd = self.pd
        if self.transition_count == 0:
            raise RuntimeError

        def normalize(probabilities: typing.List[float]) -> None:
            prob_sum = sum(probabilities)
            for i in range(len(probabilities)):
                probabilities[i] /= prob_sum

        def is_normalized(probabilities: typing.List[float]) -> bool:
            return abs(sum(probabilities) - 1.0) < 1e-10

        from_ccs_index = ColorCountState.from_state(pd, from_state).get_index(pd)
        from_ccs_samples = self.color_count_matrix.get_row_sum(from_ccs_index)
        if from_ccs_samples >= self.min_support * len(to_states):
            probabilities = [
                self.color_count_matrix.get(
                    from_ccs_index,
                    ColorCountState.from_state(pd, to_state).get_index(pd)
                ) / from_ccs_samples
                for to_state in to_states
            ]
            if not is_normalized(probabilities):
                print('debug: {} ->'.format(ColorCountState.from_index(pd, from_ccs_index)))
                for i in range(len(to_states)):
                    print('\t{:.5f} {}'.format(probabilities[i], ColorCountState.from_state(pd, to_states[i])))
            assert is_normalized(probabilities)
            return list(zip(to_states, probabilities))

        from_ils_index = InventoryLevelState.from_state(pd, from_state).get_index(pd)
        from_ils_samples = self.inventory_level_matrix.get_row_sum(from_ils_index)
        if from_ils_samples >= self.min_support * len(to_states):
            probabilities = [
                (self.color_count[to_state.color] / sum(self.color_count)) *
                self.inventory_level_matrix.get(
                    from_ils_index,
                    InventoryLevelState.from_state(pd, to_state).get_index(pd)
                ) / from_ils_samples
                for to_state in to_states
            ]
            normalize(probabilities)  # normalization required because of multiplication with color frequency
            assert is_normalized(probabilities)
            return list(zip(to_states, probabilities))

        probabilities = [
            self.color_count[to_state.color] / sum(self.color_count)
            for to_state in to_states
        ]
        normalize(probabilities)
        assert is_normalized(probabilities)
        return list(zip(to_states, probabilities))


class TransitionProbabilityMatrix:
    def __init__(self, pd: ProblemDescription, training_stats: TransitionProbabilityLearner, dtype=numpy.float64):
        self.pd: ProblemDescription = pd
        self.stats: TransitionProbabilityLearner = training_stats
        self.matrices: List[Optional[scipy.sparse.csr_matrix]] = [None] * self.pd.number_of_actions
        self.dtype = dtype

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, index: int) -> scipy.sparse.csr_matrix:
        if not isinstance(index, int):
            raise TypeError
        if not (0 <= index < len(self.matrices)):
            raise IndexError
        if self.matrices[index] is None:
            self.matrices[index] = self.get_tpm(index)
        return self.matrices[index]

    def get_tpm(self, action: int) -> scipy.sparse.csr_matrix:
        indptr = []
        indices = []
        data = []
        for state_index in range(0, self.pd.number_of_states):
            row = self.get_tpm_row(action, state_index)
            if len(row) < 1:
                raise RuntimeError
            indptr.append(len(indices))
            for cell in row:
                next_state = cell[0]
                probability = cell[1]
                indices.append(next_state)
                data.append(probability)
        indptr.append(len(indices))
        return scipy.sparse.csr_matrix(
            (numpy.array(data), numpy.array(indices), numpy.array(indptr)),
            shape=(self.pd.number_of_states, self.pd.number_of_states),
            dtype=self.dtype
        )

    def get_tpm_row(self, action: int, from_state_index: int) -> List[List[typing.Union[int, float]]]:
        """
        Return a list of lists [column_index, probability] for
        all row entries > 0.
        """
        if action > self.pd.number_of_actions:
            raise ValueError
        from_state: State = State.from_index(self.pd, from_state_index)
        if (from_state.is_invalid_state(self.pd) or
                from_state.is_invalid_request(self.pd) or
                from_state.is_invalid_action(self.pd, action)):
            return [[self.pd.invalid_state_index, 1.0]]
        to_inventory: List[int] = from_state.inventory.copy()
        if from_state.verb == 0:
            to_inventory[action] = from_state.color + 1
        else:
            to_inventory[action] = 0
        to_inventory_slots_occupied: int = sum((1 for item in to_inventory if item > 0))
        to_inventory_slots_occupied_by_color: List[int] = [
            sum((1 for item in to_inventory if item == color + 1))
            for color in range(self.pd.number_of_colors)
        ]
        to_states = []
        for to_verb in range(self.pd.number_of_verbs):
            for to_color in range(self.pd.number_of_colors):
                # don't generate output for entries with zero probability
                if to_verb == 0 and to_inventory_slots_occupied == self.pd.number_of_inventory_slots:
                    continue
                if to_verb == 1 and to_inventory_slots_occupied_by_color[to_color] == 0:
                    continue
                to_state = State(to_inventory, to_verb, to_color)
                to_states.append(to_state)
        row: List[List[typing.Union[int, float]]] = []
        for (to_state, probability) in self.stats.get_transition_probabilities(from_state, to_states):
            row.append([to_state.get_index(self.pd), probability])
        return row

    def get_transition_probability_3(self, from_state: State, to_state: State) -> float:
        """
        accurately reflects the distribution of requests in the training data
        without taking into account any sequential patterns in the requests
        -> achieves a performance similar to the greedy algorithm
        """
        occupied_slots = to_state.get_number_of_inventory_slots_occupied()
        probability = 1/6
        if to_state.verb == 0:
            free_slots = self.pd.number_of_inventory_slots - occupied_slots
            probability *= free_slots / 4
            if to_state.color == 1:
                probability *= 2
            # sums up to 1.0 for 6 free slots
        else:
            probability *= to_state.get_number_of_inventory_slots_by_color(self.pd)[to_state.color]
            # sums up to 1.0 for 6 occupied slots
        return probability

    def get_transition_probability_4(self, from_state: State, to_state: State) -> float:
        """
        like get_transition_probability_3(), but takes into account the unusually
        high frequency of store-store-store or restore-restore-restore patterns
        (still missing low probability of store-restore-store and restore-store-restore)
        """
        probability = self.get_transition_probability_3(from_state, to_state)
        if from_state.verb == to_state.verb:
            probability *= 3
        return probability

    def normalize_probabilities(self, probability_list: List[List[typing.Union[int, float]]]) -> None:
        """
        Expects a list of lists, where the second item of latter
        is a probability value. Modifies these probabilities
        in-place such that they sum up to 1.
        """
        probability_sum = sum((entry[1] for entry in probability_list))
        if probability_sum == 0.0:
            raise ValueError
        for i in range(len(probability_list)):
            probability_list[i][1] /= probability_sum
        assert abs(sum((entry[1] for entry in probability_list)) - 1.0) < 1e-10


class RewardMatrixS1:
    """
    Mimics the indexing behavior of TransitionProbabilityMatrix to
    be able to access the reward vector for a certain action.

    Turned out to be incompatible with the MDP library.
    Use RewardMatrixSA instead.
    """
    def __init__(self, pd: ProblemDescription, dtype=numpy.float32):
        self.pd: ProblemDescription = pd
        self.matrices: List[Optional[scipy.sparse.csr_matrix]] = [None] * self.pd.number_of_actions
        self.dtype = dtype

    def __len__(self):
        return len(self.pd.number_of_actions)

    def __getitem__(self, index: int) -> scipy.sparse.csr_matrix:
        if not isinstance(index, int):
            raise TypeError
        if not (0 <= index < len(self.matrices)):
            raise IndexError
        if self.matrices[index] is None:
            self.matrices[index] = self.get_sparse(index)
        return self.matrices[index]

    def get(self, action: int) -> numpy.ndarray:
        reward_matrix: numpy.ndarray = numpy.zeros(
            (self.pd.number_of_states, 1),
            dtype=self.dtype
        )
        for state_index in range(self.pd.number_of_states):
            state: State = State.from_index(self.pd, state_index)
            reward_matrix[state_index][0] = state.get_reward(self.pd, action)
        return reward_matrix

    def get_sparse(self, action: int) -> scipy.sparse.csr_matrix:
        return scipy.sparse.csr_matrix(self.get(action), dtype=self.dtype)


class RewardMatrixSA:
    """
    Provides the reward matrix in the shape of SxA.
    Use get() to retrieve it as a dense matrix.

    get_sparse() is also available but does not make much sense
    because most entries are different from zero. get_sparse()
    was introduced in the hope to solve memory issues with
    PolicyIteration and a 3x2 inventory, which did not work out.
    """
    def __init__(self, pd: ProblemDescription, dtype=numpy.int16):
        self.pd: ProblemDescription = pd
        self.dtype = dtype

    def get(self) -> numpy.ndarray:
        reward_matrix: numpy.ndarray = numpy.zeros(
            (self.pd.number_of_states, self.pd.number_of_actions),
            dtype=self.dtype
        )
        for action in range(self.pd.number_of_actions):
            for state_index in range(self.pd.number_of_states):
                state: State = State.from_index(self.pd, state_index)
                reward_matrix[state_index][action] = state.get_reward(self.pd, action)
        return reward_matrix

    def get_sparse(self) -> scipy.sparse.csr_matrix:
        return scipy.sparse.csr_matrix(self.get(), dtype=self.dtype)
