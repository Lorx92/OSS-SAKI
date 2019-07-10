import typing


class ProblemDescription:
    """
    Contains all information specific to a certain problem,
    e.g. inventory shape, number of states, ...

    Can also compute the Manhattan Distance between inventory slots.
    Can also iterate over all valid states or simplifications thereof.
    """
    def __init__(self, inventory_cols: int, inventory_rows: int, colors: typing.List[str] = None,
                 verbs: typing.List[str] = None, with_ignore_request_action: bool = True):
        self.number_of_inventory_cols = inventory_cols
        self.number_of_inventory_rows = inventory_rows
        self.number_of_inventory_slots = self.number_of_inventory_cols * self.number_of_inventory_rows
        self.with_ignore_request_action = with_ignore_request_action
        self.number_of_actions = self.number_of_inventory_slots
        if self.with_ignore_request_action:
            self.number_of_actions += 1
        if colors is None:
            self.colors = ['blue', 'red', 'white']
        else:
            self.colors = colors
        self.number_of_colors = len(self.colors)
        if verbs is None:
            self.verbs = ['store', 'restore']
        else:
            self.verbs = verbs
        self.number_of_verbs = len(self.verbs)
        self.number_of_inventory_slot_states = self.number_of_colors + 1
        self.number_of_states = ((self.number_of_inventory_slot_states ** self.number_of_inventory_slots) *
                                 self.number_of_verbs * self.number_of_colors)
        self.invalid_state_index = self.number_of_states
        self.number_of_states += 1
        self.invalid_state = State([0] * self.number_of_inventory_slots, 0, 0, index=self.invalid_state_index)

        self.simplified_inventories_by_index = list(ColorCountState.iterate_over_simplified_inventories(
            self.number_of_inventory_slots, self.number_of_colors))
        self.simplified_inventory_string_to_index = {
            ColorCountState.simplified_inventory_as_string(inv): index
            for (index, inv) in enumerate(self.simplified_inventories_by_index)
        }
        self.number_of_color_count_states = (len(self.simplified_inventories_by_index) *
                                             self.number_of_verbs * self.number_of_colors)
        self.invalid_color_count_state_index = self.number_of_color_count_states
        self.number_of_color_count_states += 1
        self.invalid_color_count_state = ColorCountState(
            [0] * self.number_of_colors, 0, 0, index=self.invalid_color_count_state_index)

        self.number_of_inventory_level_states = (self.number_of_inventory_slots + 1) * self.number_of_verbs
        self.invalid_inventory_level_state_index = self.number_of_inventory_level_states
        self.number_of_inventory_level_states += 1
        self.invalid_inventory_level_state = InventoryLevelState(0, 0, index=self.invalid_inventory_level_state_index)

    def __repr__(self):
        return 'ProblemDescription({}, {})'.format(self.number_of_inventory_cols, self.number_of_inventory_rows)

    def get_manhattan_distance_to_last_inventory_slot(self, inventory_slot: int) -> int:
        # zero-based row and column numbers for the inventory slot defined by the action number
        # example for 3x2 inventory:
        # (0,0) (0,1) (0,2) for action numbers 0  1  2
        # (1,0) (1,1) (1,2)                    3  4  5
        (slot_row, slot_col) = divmod(inventory_slot, self.number_of_inventory_cols)
        dist_x = self.number_of_inventory_cols - 1 - slot_col
        dist_y = self.number_of_inventory_rows - 1 - slot_row
        assert 0 <= dist_x < self.number_of_inventory_cols
        assert 0 <= dist_y < self.number_of_inventory_rows
        distance = dist_x + dist_y
        return distance

    def get_valid_state_transitions(self):
        n_o_actions = self.number_of_actions
        if self.with_ignore_request_action:
            n_o_actions -= 1
        for si1 in range(self.number_of_states - 1):
            s1 = State.from_index(self, si1)
            if s1.is_invalid_request(self):
                continue
            for action in range(n_o_actions):
                if s1.is_invalid_action(self, action):
                    continue
                for verb in range(self.number_of_verbs):
                    for color in range(self.number_of_colors):
                        s2 = s1.apply_action(self, action, verb, color)
                        if s2.is_invalid_state(self) or s2.is_invalid_request(self):
                            continue
                        yield (s1.get_index(self), s2.get_index(self))

    def get_valid_color_count_state_transitions(self):
        action = 0
        for si1 in range(self.number_of_color_count_states - 1):
            s1 = ColorCountState.from_index(self, si1)
            if s1.is_invalid_request(self):
                continue
            for verb in range(self.number_of_verbs):
                for color in range(self.number_of_colors):
                    s2 = s1.apply_action(self, action, verb, color)
                    if s2.is_invalid_color_count_state(self) or s2.is_invalid_request(self):
                        continue
                    yield (s1.get_index(self), s2.get_index(self))

    def get_valid_inventory_level_state_transitions(self):
        action = 0
        for si1 in range(self.number_of_inventory_level_states - 1):
            s1 = InventoryLevelState.from_index(self, si1)
            if s1.is_invalid_request(self):
                continue
            for verb in range(self.number_of_verbs):
                s2 = s1.apply_action(self, action, verb)
                if s2.is_invalid_inventory_level(self) or s2.is_invalid_request(self):
                    continue
                yield (s1.get_index(self), s2.get_index(self))


class State:
    """
    Describes a state.
    """
    def __init__(self, inventory: typing.List[int], verb: int, color: int, index: typing.Optional[int] = None):
        self.inventory = inventory
        self.verb = verb
        self.color = color
        self._index = index
        self._is_invalid_request = None
        self._number_of_inventory_slots_occupied = None
        self._number_of_inventory_slots_by_color = None

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        if self._index is not None and other._index is not None:
            return self._index == other._index
        return self.inventory == other.inventory and self.verb == other.verb and self.color == other.color

    def __repr__(self):
        return 'State([' + ','.join((str(item) for item in self.inventory)) + '], {}, {}, index={})'.format(
            self.verb, self.color, self._index)

    @staticmethod
    def from_index(pd: ProblemDescription, index: int) -> 'State':
        """
        Factory method to create a state from its index value.
        """
        assert 0 <= index <= pd.invalid_state_index
        if index == pd.invalid_state_index:
            return pd.invalid_state
        index2 = index
        (index2, color) = divmod(index2, pd.number_of_colors)
        (index2, verb) = divmod(index2, pd.number_of_verbs)
        inventory = [0] * pd.number_of_inventory_slots
        for i in range(pd.number_of_inventory_slots):
            (index2, item) = divmod(index2, pd.number_of_inventory_slot_states)
            inventory[i] = item
        assert index2 == 0
        return State(inventory, verb, color, index=index)

    def get_index(self, pd: ProblemDescription) -> int:
        if self._index is None:
            index = 0
            assert len(self.inventory) == pd.number_of_inventory_slots
            for (i, item) in enumerate(self.inventory):
                index += (pd.number_of_inventory_slot_states ** i) * item
                assert 0 <= item < pd.number_of_inventory_slot_states
            index *= pd.number_of_verbs
            index += self.verb
            assert self.verb < pd.number_of_verbs
            index *= pd.number_of_colors
            index += self.color
            assert self.color < pd.number_of_colors
            assert index < pd.invalid_state_index
            self._index = index
        return self._index

    def get_reward(self, pd: ProblemDescription, action: int) -> int:
        # typical reward r: -(#rows - 1 + #cols - 1) <= r <= 0
        # if in invalid state: r == -(#rows * #cols)
        # (and the only successor state to the invalid state is the invalid state itself)
        # if invalid action chosen: r == -2(#rows * #cols)
        # (and the successor state is the invalid state)
        if self.is_invalid_action(pd, action):
            return -pd.number_of_inventory_slots * 2
        if self.is_invalid_state(pd):
            return -pd.number_of_inventory_slots
        if pd.with_ignore_request_action and action == pd.number_of_actions - 1:
            return -pd.number_of_inventory_slots
        return -pd.get_manhattan_distance_to_last_inventory_slot(action)

    def is_invalid_request(self, pd: ProblemDescription) -> bool:
        if self._is_invalid_request is None:
            if self.get_index(pd) == pd.invalid_state_index:
                self._is_invalid_request = False
            elif self.verb == 0:  # store
                self._is_invalid_request = self.get_number_of_inventory_slots_occupied() == pd.number_of_inventory_slots
            else:  # restore
                self._is_invalid_request = self.get_number_of_inventory_slots_by_color(pd)[self.color] == 0
        return self._is_invalid_request

    def is_invalid_state(self, pd: ProblemDescription) -> bool:
        return self._index == pd.invalid_state_index

    def apply_action(self, pd: ProblemDescription, action: int,
                     next_verb: typing.Optional[int] = None, next_color: typing.Optional[int] = None
                     ) -> 'State':
        assert 0 <= action < pd.number_of_actions
        if self.is_invalid_state(pd):
            return pd.invalid_state
        if pd.with_ignore_request_action and action == pd.number_of_actions - 1:
            return pd.invalid_state
        assert action < pd.number_of_inventory_slots
        if self.verb == 0 and self.inventory[action] != 0:
            return pd.invalid_state
        if self.verb == 1 and self.inventory[action] != self.color + 1:
            return pd.invalid_state
        modified_inventory = self.inventory.copy()
        if self.verb == 0:
            modified_inventory[action] = self.color + 1
        else:
            modified_inventory[action] = 0
        if next_verb is None:
            next_verb = 0  # caller needs to fill in next request itself
        if next_color is None:
            next_color = 0
        return State(modified_inventory, next_verb, next_color)

    def is_invalid_action(self, pd: ProblemDescription, action: int) -> bool:
        assert 0 <= action < pd.number_of_actions
        if self.is_invalid_state(pd):
            if pd.with_ignore_request_action:
                # should ignore the request if in invalid state
                return action < pd.number_of_actions - 1
            else:
                return False
        if pd.with_ignore_request_action:
            if action == pd.number_of_actions - 1:  # ignore the request
                if self.verb == 0:
                    # if item could be stored but request was ignored instead -> invalid action
                    return self.can_store_item()
                else:
                    # if item could be restored but request was ignored instead -> invalid action
                    return self.can_restore_item(pd)
            else:  # execute the request
                if self.verb == 0:
                    # if item can't be stored (inventory full) -> invalid action
                    if not self.can_store_item():
                        return True
                    # if chosen inventory slot wasn't empty -> invalid action
                    if self.inventory[action] > 0:
                        return True
                    return False
                else:
                    # if item can't be restored (does not exist in inventory) -> invalid action
                    if not self.can_restore_item(pd):
                        return True
                    # if chosen inventory slot didn't contain the proper item type -> invalid action
                    if self.inventory[action] != self.color + 1:
                        return True
                    return False
        else:
            if self.verb == 0:
                if not self.can_store_item():
                    # if item can't be stored (inventory full) -> valid action
                    return False
                if self.inventory[action] != 0:
                    # item can be stored but chosen inventory slot isn't empty -> invalid action
                    return True
                return False
            else:
                if not self.can_restore_item(pd):
                    # if item can't be restored (does not exist in inventory) -> valid action
                    return False
                if self.inventory[action] != self.color + 1:
                    # if chosen inventory slot didn't contain the proper item type -> invalid action
                    return True
                return False

    def get_number_of_inventory_slots_occupied(self) -> int:
        if self._number_of_inventory_slots_occupied is None:
            self._number_of_inventory_slots_occupied = sum((1 for item in self.inventory if item > 0))
        return self._number_of_inventory_slots_occupied

    def get_number_of_inventory_slots_by_color(self, pd: ProblemDescription) -> typing.List[int]:
        if self._number_of_inventory_slots_by_color is None:
            self._number_of_inventory_slots_by_color = [
                sum((1 for item in self.inventory if item == color + 1))
                for color in range(pd.number_of_colors)
            ]
        return self._number_of_inventory_slots_by_color

    def can_store_item(self) -> bool:
        return self.get_number_of_inventory_slots_occupied() < len(self.inventory)

    def can_restore_item(self, pd: ProblemDescription) -> bool:
        return self.get_number_of_inventory_slots_by_color(pd)[self.color] > 0


class ColorCountState:
    """
    A simplified version of state where item order is ignored.
    Used in TransitionProbabilityLearner for counting observed state transitions.
    """
    def __init__(self, number_of_inventory_slots_by_color: typing.List[int],
                 verb: int, color: int, index: typing.Optional[int] = None):
        self.number_of_inventory_slots_by_color = number_of_inventory_slots_by_color
        self.verb = verb
        self.color = color
        self._index = index
        self._is_invalid_request = None
        self._number_of_inventory_slots_occupied = None

    def __eq__(self, other):
        if not isinstance(other, ColorCountState):
            return False
        if self._index is not None and other._index is not None:
            return self._index == other._index
        return (self.number_of_inventory_slots_by_color == other.number_of_inventory_slots_by_color and
                self.verb == other.verb and self.color == other.color)

    def __repr__(self):
        return 'ColorCountState([' + ','.join(
            (str(color_count) for color_count in self.number_of_inventory_slots_by_color)
        ) + '], {}, {}, index={})'.format(self.verb, self.color, self._index)

    @staticmethod
    def iterate_over_simplified_inventories(number_of_inventory_slots, number_of_colors, current_color=0):
        """
        a recursive iterator over all inventory combinations
        """
        for color_count in range(number_of_inventory_slots + 1):
            if current_color == number_of_colors - 1:
                # innermost loop
                # -> don't recurse, just return color count for current color as a list
                yield [color_count]
            else:
                # recursive call for remaining colors
                for other_color_counts in ColorCountState.iterate_over_simplified_inventories(
                        number_of_inventory_slots, number_of_colors, current_color=current_color + 1):
                    if current_color > 0:
                        # neither innermost nor outermost loop
                        # -> concatenate color count for current color with list from inner loops
                        intermediate_color_counts = [color_count]
                        intermediate_color_counts.extend(other_color_counts)
                        yield intermediate_color_counts
                    else:
                        # outermost loop, current_color == 0
                        # -> concatenate color count for current color with list from inner loops
                        # and return final list if sum of color counts is within proper bounds
                        final_color_counts = [color_count]
                        final_color_counts.extend(other_color_counts)
                        if sum(final_color_counts) <= number_of_inventory_slots:
                            yield final_color_counts

    @staticmethod
    def simplified_inventory_as_string(number_of_inventory_slots_by_color: typing.List[int]):
        return '[' + ','.join((str(count) for count in number_of_inventory_slots_by_color)) + ']'

    @staticmethod
    def from_index(pd: ProblemDescription, index: int) -> 'ColorCountState':
        assert 0 <= index <= pd.invalid_color_count_state_index
        if index == pd.invalid_color_count_state_index:
            return pd.invalid_color_count_state
        index2 = index
        (index2, color) = divmod(index2, pd.number_of_colors)
        (index2, verb) = divmod(index2, pd.number_of_verbs)
        (index2, simplified_inventory_index) = divmod(index2, pd.number_of_color_count_states)
        number_of_inventory_slots_by_color = pd.simplified_inventories_by_index[simplified_inventory_index].copy()
        assert index2 == 0
        return ColorCountState(number_of_inventory_slots_by_color, verb, color, index=index)

    @staticmethod
    def from_state(pd: ProblemDescription, state: State) -> 'ColorCountState':
        if state.is_invalid_state(pd):
            return pd.invalid_color_count_state
        return ColorCountState(state.get_number_of_inventory_slots_by_color(pd), state.verb, state.color)

    def get_index(self, pd: ProblemDescription) -> int:
        if self._index is None:
            assert len(self.number_of_inventory_slots_by_color) == pd.number_of_colors
            index = pd.simplified_inventory_string_to_index[
                ColorCountState.simplified_inventory_as_string(self.number_of_inventory_slots_by_color)]
            index *= pd.number_of_verbs
            index += self.verb
            assert self.verb < pd.number_of_verbs
            index *= pd.number_of_colors
            index += self.color
            assert self.color < pd.number_of_colors
            assert index < pd.invalid_color_count_state_index
            self._index = index
        return self._index

    def is_invalid_request(self, pd: ProblemDescription) -> bool:
        if self._is_invalid_request is None:
            if self.get_index(pd) == pd.invalid_color_count_state_index:
                self._is_invalid_request = False
            elif self.verb == 0:  # store
                self._is_invalid_request = self.get_number_of_inventory_slots_occupied() == pd.number_of_inventory_slots
            else:  # restore
                self._is_invalid_request = self.number_of_inventory_slots_by_color[self.color] == 0
        return self._is_invalid_request

    def is_invalid_color_count_state(self, pd: ProblemDescription) -> bool:
        return self._index == pd.invalid_color_count_state_index

    def apply_action(self, pd: ProblemDescription, action: int,
                     next_verb: typing.Optional[int] = None, next_color: typing.Optional[int] = None
                     ) -> 'ColorCountState':
        assert 0 <= action < pd.number_of_actions
        if self.is_invalid_color_count_state(pd):
            return pd.invalid_color_count_state
        if pd.with_ignore_request_action and action == pd.number_of_actions - 1:
            return pd.invalid_color_count_state
        assert action < pd.number_of_inventory_slots
        if self.verb == 0 and sum(self.number_of_inventory_slots_by_color) == pd.number_of_inventory_slots:
            return pd.invalid_color_count_state
        if self.verb == 1 and self.number_of_inventory_slots_by_color[self.color] == 0:
            return pd.invalid_color_count_state
        modified_simplified_inventory = self.number_of_inventory_slots_by_color.copy()
        if self.verb == 0:
            modified_simplified_inventory[self.color] += 1
        else:
            modified_simplified_inventory[self.color] -= 1
        assert 0 <= sum(self.number_of_inventory_slots_by_color) <= pd.number_of_inventory_slots
        if next_verb is None:
            next_verb = 0  # caller needs to fill in next request itself
        if next_color is None:
            next_color = 0
        return ColorCountState(modified_simplified_inventory, next_verb, next_color)

    def is_invalid_action(self, pd: ProblemDescription, action: int) -> bool:
        assert 0 <= action < pd.number_of_actions
        if self.is_invalid_color_count_state(pd):
            if pd.with_ignore_request_action:
                # should ignore the request if in invalid state
                return action < pd.number_of_actions - 1
            else:
                return False
        if pd.with_ignore_request_action:
            if action == pd.number_of_actions - 1:  # ignore the request
                if self.verb == 0:
                    # if item could be stored but request was ignored instead -> invalid action
                    return self.can_store_item(pd)
                else:
                    # if item could be restored but request was ignored instead -> invalid action
                    return self.can_restore_item()
            else:  # execute the request
                if self.verb == 0:
                    # if item can't be stored (inventory full) -> invalid action
                    return not self.can_store_item(pd)
                else:
                    # if item can't be restored (does not exist in inventory) -> invalid action
                    return not self.can_restore_item()
        return False

    def get_number_of_inventory_slots_occupied(self):
        if self._number_of_inventory_slots_occupied is None:
            self._number_of_inventory_slots_occupied = sum(self.number_of_inventory_slots_by_color)
        return self._number_of_inventory_slots_occupied

    def can_store_item(self, pd: ProblemDescription) -> bool:
        return self.get_number_of_inventory_slots_occupied() < pd.number_of_inventory_slots

    def can_restore_item(self) -> bool:
        return self.number_of_inventory_slots_by_color[self.color] > 0


class InventoryLevelState:
    """
    Even more simplified version of state where not only
    item order is ignored but even color as well.

    Used in TransitionProbabilityLearner for counting observed state transitions.
    """
    def __init__(self, inventory_slots_occupied: int,
                 verb: int, index: typing.Optional[int] = None):
        self.inventory_slots_occupied = inventory_slots_occupied
        self.verb = verb
        self._index = index
        self._is_invalid_request = None

    def __eq__(self, other):
        if not isinstance(other, InventoryLevelState):
            return False
        if self._index is not None and other._index is not None:
            return self._index == other._index
        return (self.inventory_slots_occupied == other.inventory_slots_occupied and
                self.verb == other.verb)

    def __repr__(self):
        return 'InventoryLevelState({}, {}, index={})'.format(
            self.inventory_slots_occupied, self.verb, self._index)

    @staticmethod
    def from_index(pd: ProblemDescription, index: int) -> 'InventoryLevelState':
        assert 0 <= index < pd.number_of_inventory_level_states
        if index == pd.invalid_inventory_level_state_index:
            return pd.invalid_inventory_level_state
        index2 = index
        (index2, verb) = divmod(index2, pd.number_of_verbs)
        (index2, inventory_slots_occupied) = divmod(index2, pd.number_of_inventory_level_states)
        assert index2 == 0
        return InventoryLevelState(inventory_slots_occupied, verb, index=index)

    @staticmethod
    def from_state(pd: ProblemDescription, state: State) -> 'InventoryLevelState':
        if state.is_invalid_state(pd):
            return pd.invalid_inventory_level_state
        return InventoryLevelState(state.get_number_of_inventory_slots_occupied(), state.verb)

    def get_index(self, pd: ProblemDescription) -> int:
        if self._index is None:
            index = self.inventory_slots_occupied
            index *= pd.number_of_verbs
            index += self.verb
            assert self.verb < pd.number_of_verbs
            assert index < pd.invalid_inventory_level_state_index
            self._index = index
        return self._index

    def is_invalid_request(self, pd: ProblemDescription) -> bool:
        if self._is_invalid_request is None:
            if self.get_index(pd) == pd.invalid_inventory_level_state_index:
                self._is_invalid_request = False
            elif self.verb == 0:  # store
                self._is_invalid_request = self.inventory_slots_occupied == pd.number_of_inventory_slots
            else:  # restore
                self._is_invalid_request = self.inventory_slots_occupied == 0
        return self._is_invalid_request

    def is_invalid_inventory_level(self, pd: ProblemDescription) -> bool:
        return self._index == pd.invalid_inventory_level_state_index

    def apply_action(self, pd: ProblemDescription, action: int,
                     next_verb: typing.Optional[int] = None) -> 'InventoryLevelState':
        assert 0 <= action < pd.number_of_actions
        if self.is_invalid_inventory_level(pd):
            return pd.invalid_inventory_level_state
        if pd.with_ignore_request_action and action == pd.number_of_actions - 1:
            return pd.invalid_inventory_level_state
        assert action < pd.number_of_inventory_slots
        if self.verb == 0 and self.inventory_slots_occupied == pd.number_of_inventory_slots:
            return pd.invalid_inventory_level_state
        if self.verb == 1 and self.inventory_slots_occupied == 0:
            return pd.invalid_inventory_level_state
        modified_inventory_level = self.inventory_slots_occupied
        if self.verb == 0:
            modified_inventory_level += 1
        else:
            modified_inventory_level -= 1
        assert 0 <= modified_inventory_level <= pd.number_of_inventory_slots
        if next_verb is None:
            next_verb = 0  # caller needs to fill in next request itself
        return InventoryLevelState(modified_inventory_level, next_verb)

    def is_invalid_action(self, pd: ProblemDescription, action: int) -> bool:
        assert 0 <= action < pd.number_of_actions
        if self.is_invalid_inventory_level(pd):
            if pd.with_ignore_request_action:
                # should ignore the request if in invalid state
                return action < pd.number_of_actions - 1
            else:
                return False
        if pd.with_ignore_request_action:
            if action == pd.number_of_actions - 1:  # ignore the request
                if self.verb == 0:
                    # if item could be stored but request was ignored instead -> invalid action
                    return self.can_store_item(pd)
                else:
                    # if item could be restored but request was ignored instead -> invalid action
                    return self.can_restore_item()
            else:  # execute the request
                if self.verb == 0:
                    # if item can't be stored (inventory full) -> invalid action
                    return not self.can_store_item(pd)
                else:
                    # if item can't be restored (does not exist in inventory) -> invalid action
                    return not self.can_restore_item()
        return False

    def get_number_of_inventory_slots_occupied(self):
        return self.inventory_slots_occupied

    def can_store_item(self, pd: ProblemDescription) -> bool:
        return self.inventory_slots_occupied < pd.number_of_inventory_slots

    def can_restore_item(self) -> bool:
        return self.inventory_slots_occupied > 0
