from typing import Union
from dh.StateExpertsOnly import StateExpertsOnly
from dh.StateExpertsCashShares import StateExpertsCashShares


class Experience:
    """
    Encapsulates a state transition and the resulting reward.
    """
    def __init__(self,
                 state1: Union[StateExpertsOnly, StateExpertsCashShares] = None,
                 actions=None,
                 reward=None,
                 state2: Union[StateExpertsOnly, StateExpertsCashShares] = None):
        self.state1: Union[StateExpertsOnly, StateExpertsCashShares] = state1
        self.actions = actions
        self.reward = reward
        self.state2: Union[StateExpertsOnly, StateExpertsCashShares] = state2

    def get_portfolio_rel_immediate_change(self):
        return self.state2.portfolio_value / self.state1.portfolio_value

    def get_portfolio_abs_immediate_change(self):
        return self.state2.portfolio_value - self.state1.portfolio_value
