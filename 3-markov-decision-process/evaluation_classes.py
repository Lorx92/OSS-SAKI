from data_exploration_classes import Data
from problem_description_and_state import ProblemDescription, State
from training_classes import TransitionProbabilityMatrix

import mdptoolbox
import typing
from typing import List, Optional


class GreedyPolicy:
    """
    Returns the action for a given state that uses the closest
    inventory slot suitable for the request given by the state.

    Use with the Evaluation class to determine a score and compare
    it with other policies.
    """
    def __init__(self, pd: ProblemDescription):
        self.pd = pd
        self.inventory_slots_by_distance = sorted(
            list(range(self.pd.number_of_inventory_slots)),
            key=self.pd.get_manhattan_distance_to_last_inventory_slot
        )

    def get_action(self, state_index: int) -> int:
        pd = self.pd
        state: State = State.from_index(pd, state_index)
        if state.is_invalid_state(pd) or state.is_invalid_request(pd):
            if pd.with_ignore_request_action:
                return pd.number_of_actions - 1
            else:
                return 0
        if state.verb == 0:
            for slot in self.inventory_slots_by_distance:
                if state.inventory[slot] == 0:
                    return slot
            # finding no free slot should have been catched above already by state.is_invalid_request
            raise RuntimeError
        else:
            for slot in self.inventory_slots_by_distance:
                if state.inventory[slot] == state.color + 1:
                    return slot
            # not finding an item with that color should have been catched above already
            raise RuntimeError


class PolicyFromMDPSolver:
    """
    Wraps a policy as given by a solver from the mdptoolbox library.
    Mimics the behavior of GreedyPolicy.

    Use with the Evaluation class to determine a score and compare
    it with other policies.
    """
    def __init__(self, pd: ProblemDescription, solver: mdptoolbox.mdp.MDP):
        if not isinstance(solver, mdptoolbox.mdp.MDP):
            raise TypeError
        self.pd = pd
        self.solver = solver
        if self.solver.time is None:
            raise ValueError('solver has not run')
        assert len(self.solver.policy) == self.pd.number_of_states

    def get_action(self, state_index: int) -> int:
        assert 0 <= state_index < self.pd.number_of_states
        return self.solver.policy[state_index]


class Evaluation:
    """
    Used to evaluate a given policy over a sequence of requests in
    the evaluation data.

    A TransitionProbabilityMatrix can be passed for debug purposes.
    A warning will be printed when a transition occurs which, according
    to the matrix, has a probability of zero.
    """
    def __init__(self, pd: ProblemDescription, evaluation_data: Data,
                 policy: typing.Union[PolicyFromMDPSolver, GreedyPolicy],
                 tpm: Optional[TransitionProbabilityMatrix] = None,
                 transition_probability_warn_limit=0.0):
        if not isinstance(policy, (PolicyFromMDPSolver, GreedyPolicy)):
            raise TypeError
        self.pd = pd
        self.evaluation_data = evaluation_data
        self.policy = policy
        self.tpm = tpm
        self.transition_probability_warn_limit = transition_probability_warn_limit
        self._total_reward = None
        self._score = None

    def get_total_reward(self) -> int:
        """
        Returns the total reward given for applying the policy to the
        evaluation data. The returned number will be in the range
        (-infty, 0]. It will depend on the number of requests in the evaluation
        data.
        """
        if self._total_reward is None:
            pd = self.pd
            self._total_reward = 0
            current_state = None
            next_state = State([0] * pd.number_of_inventory_slots, 0, 0)
            for (verb, color) in self.evaluation_data.requests:
                if not next_state.is_invalid_state(pd):
                    next_state = State(next_state.inventory, verb, color)
                if self.tpm is not None and current_state is not None:
                    prob = self.tpm.get_transition_probability(current_state, next_state)
                    if prob <= self.transition_probability_warn_limit:
                        print('debug: {} -> {} with prob. {:.7f}'.format(repr(current_state), repr(next_state), prob))
                current_state = next_state
                action = self.policy.get_action(current_state.get_index(pd))
                self._total_reward += current_state.get_reward(pd, action)
                # update inventory (need to update verb and color later when we know what values they have)
                next_state = current_state.apply_action(pd, action)
        return self._total_reward

    def get_score(self) -> float:
        """
        Returns a score in the range of [0.0, 1.0] as measure of how good the
        policy has performed on the evaluation data. 1.0 means perfect,
        0.0 means worst possible performance under normal conditions (i.e. not
        counting invalid states, requests or actions).
        """
        if self._score is None:
            pd = self.pd
            avg_reward_per_request = self.get_total_reward() / len(self.evaluation_data.requests)
            min_typical_reward = min((
                -pd.get_manhattan_distance_to_last_inventory_slot(slot)
                for slot in range(pd.number_of_inventory_slots)
            ))
            normalized_avg_reward_per_request = avg_reward_per_request / -min_typical_reward
            # now from [-1.0, 0.0] if no invalid stuff occurred
            normalized_avg_reward_per_request += 1.0
            # now from [0.0, 1.0]
            # cut off any negative values due to invalid stuff
            normalized_avg_reward_per_request = max(0.0, normalized_avg_reward_per_request)
            self._score = normalized_avg_reward_per_request
        return self._score
