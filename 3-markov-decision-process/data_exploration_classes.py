from problem_description_and_state import ProblemDescription

import collections
import random
import typing


class Data:
    """
    Encapsulates reading and validating input data from a text file.
    Uses the Statistics class to do a statistical analysis of the data.
    """

    def __init__(self, pd: ProblemDescription, filepath: str):
        self.pd = pd
        self.filepath = filepath
        self.requests = None
        self.read_requests_from_file()

    def read_requests_from_file(self) -> None:
        """
        Initializes this Data instance by reading requests line by line
        from the text file given by filepath.

        Each line in the text file should represent one request, consisting
        of a verb ("store" or "restore") followed by the type of the item
        ("blue", "red", "white").

        Comment lines (beginning with "#") or empty lines are ignored.

        Requests are stored internally as a list of verb-color tuples
        (each tuple describes one request).
        Verb is an int (index into pd.verbs).
        Color is an int (index into pd.colors).

        Raises a ValueError if a line contains illegal input.
        """
        self.requests = []
        with open(self.filepath, 'r') as input_file:
            for (line_number, line) in enumerate(input_file, start=1):
                line = line.strip()
                if len(line) == 0 or line.startswith('#'):
                    continue
                verb = None
                for (verb_index, verb_label) in enumerate(self.pd.verbs):
                    if line.startswith(verb_label):
                        verb = verb_index
                        break
                if verb is None:
                    self.process_illegal_input_line(line, line_number)
                rest_of_line = line[len(self.pd.verbs[verb]):].lstrip()
                color = None
                for (color_index, color_label) in enumerate(self.pd.colors):
                    if rest_of_line == color_label:
                        color = color_index
                        break
                if color is None:
                    self.process_illegal_input_line(line, line_number)
                self.requests.append((verb, color))

    def process_illegal_input_line(self, line, line_number):
        """
        error handling helper method
        """
        error_msg = 'found illegal input line (#{}) in "{}":\n{}'.format(line_number, self.filepath, line)
        raise ValueError(error_msg)

    def print_statistics(self):
        """
        prints results of statistical analysis of this data
        """
        print(self.filepath)
        print('-----------------------------------------------------------')
        stats = Statistics(self)
        stats.do_basic_analysis()
        stats.print_basic_counts()


class Statistics:
    """
    Used by the Data class to run a statistical analysis on the data.

    Parameters
    ----------
    data
        A Data instance to run the analysis on.
    """

    def __init__(self, data: typing.Union[Data, 'GeneratedData']):
        self.data = data
        pd = self.data.pd
        self.verb_count_per_inventory_level = [[0, 0] for _ in range(pd.number_of_inventory_slots + 1)]
        self.repeated_verb_count_per_inventory_level = [[0, 0] for _ in range(pd.number_of_inventory_slots + 1)]
        self.alternating_verb_count_per_inventory_level = [[0, 0] for _ in range(pd.number_of_inventory_slots + 1)]
        self.counts_per_color = {
            key: ([0] * pd.number_of_colors)
            for key in ('request', 'min', 'max', 'avg', 'final')
        }
        self.counts = {
            key: 0
            for key in ('store', 'restore', 'min', 'max', 'avg', 'final', 'repeated_verb')
        }
        self.request_shares_per_color = [0.0] * pd.number_of_colors
        self.request_shares_per_inventory_level_and_verb = [
            [
                0.0
                for _ in range(pd.number_of_verbs)
            ]
            for _ in range(pd.number_of_inventory_slots + 1)
        ]
        self.request_shares_per_inventory_level_and_verb1_and_verb2 = [
            [
                [
                    0 for _ in range(pd.number_of_verbs)  # verb2
                ]
                for _ in range(pd.number_of_verbs)  # verb1
            ]
            for _ in range(pd.number_of_inventory_slots + 1)
        ]

    def do_basic_analysis(self):
        """
        computes the analysis results
        """

        cpc = self.counts_per_color
        c = self.counts
        pd = self.data.pd
        last_verb = None
        for (verb, color) in self.data.requests:
            cpc['request'][color] += 1
            if last_verb is not None:
                v1 = last_verb
                v2 = verb
                self.request_shares_per_inventory_level_and_verb1_and_verb2[c['final']][v1][v2] += 1
                if v1 == v2:
                    c['repeated_verb'] += 1
            self.verb_count_per_inventory_level[c['final']][verb] += 1
            if verb == 0:  # store
                c['store'] += 1
                c['final'] += 1
                if c['final'] > c['max']:
                    c['max'] = c['final']
                cpc['final'][color] += 1
                if cpc['final'][color] > cpc['max'][color]:
                    cpc['max'][color] = cpc['final'][color]
            else:  # restore
                c['restore'] += 1
                c['final'] -= 1
                if c['final'] < c['min']:
                    c['min'] = c['final']
                cpc['final'][color] -= 1
                if cpc['final'][color] < cpc['min'][color]:
                    cpc['min'][color] = cpc['final'][color]
            c['avg'] += c['final']
            for color2 in range(pd.number_of_colors):
                cpc['avg'][color2] += cpc['final'][color2]
            last_verb = verb
        c['avg'] /= len(self.data.requests)
        for color in range(len(self.request_shares_per_color)):
            self.request_shares_per_color[color] = cpc['request'][color] / len(self.data.requests)
            cpc['avg'][color] /= len(self.data.requests)
        for level in range(len(self.request_shares_per_inventory_level_and_verb1_and_verb2)):
            for v1 in range(pd.number_of_verbs):
                self.request_shares_per_inventory_level_and_verb[level][v1] = (
                    self.verb_count_per_inventory_level[level][v1] / len(self.data.requests)
                )
                for v2 in range(pd.number_of_verbs):
                    self.request_shares_per_inventory_level_and_verb1_and_verb2[level][v1][v2] /= (
                        len(self.data.requests)
                    )

    def print_basic_counts(self):
        """
        prints the results from do_basic_analysis()
        """
        cpc = self.counts_per_color
        c = self.counts
        pd = self.data.pd
        print('{} requests total'.format(len(self.data.requests)))
        print('{} store requests, {} restore requests'.format(c['store'], c['restore']))
        print('{} requests ({:.2f}) with the same verb as the one before'.format(
            c['repeated_verb'], c['repeated_verb'] / len(self.data.requests)
        ))
        print('min {min} / max {max} / avg {avg:.2f} / final {final}'.format(
            min=c['min'], max=c['max'], avg=c['avg'], final=c['final']
        ))
        for (color_index, color_label) in enumerate(pd.colors):
            print(('{color: <5} min {min} / max {max} / avg {avg:.2f} / final {final} / '
                   'request share {request_share:.2f}').format(
                color=color_label,
                min=cpc['min'][color_index],
                max=cpc['max'][color_index],
                avg=cpc['avg'][color_index],
                final=cpc['final'][color_index],
                request_share=cpc['request'][color_index] / len(self.data.requests)
            ))
        for (inventory_level, verb_counts) in enumerate(self.verb_count_per_inventory_level):
            print('at {} items: {:.3f} store / {:.3f} restore'.format(
                inventory_level,
                verb_counts[0] / len(self.data.requests),
                verb_counts[1] / len(self.data.requests)
            ))
            print('            {:.3f} store after store / {:.3f} restore after restore'.format(
                self.request_shares_per_inventory_level_and_verb1_and_verb2[inventory_level][0][0],
                self.request_shares_per_inventory_level_and_verb1_and_verb2[inventory_level][1][1]
            ))
            print('            {:.3f} store after restore / {:.3f} restore after store'.format(
                self.request_shares_per_inventory_level_and_verb1_and_verb2[inventory_level][1][0],
                self.request_shares_per_inventory_level_and_verb1_and_verb2[inventory_level][0][1]
            ))


class RequestGenerator:
    """
    mimics the request characteristics found in the data files
    (except sequential request patterns)
    """
    def __init__(self, pd: ProblemDescription, limit: int = 1000000):
        self.pd = pd
        self.limit = limit
        self.seed = random.randrange(int(1e100))

    def __iter__(self):
        return self.iterate()

    def iterate(self):
        rand_gen = random.Random(self.seed)
        pd = self.pd
        inventory = [0] * pd.number_of_inventory_slots
        for _ in range(self.limit):
            index = rand_gen.randrange(pd.number_of_inventory_slots)
            verb = 0 if inventory[index] == 0 else 1
            if verb == 0:
                color = rand_gen.randrange(pd.number_of_colors + 1)
                if color > 1:
                    color -= 1
                inventory[index] = color + 1
            else:
                color = inventory[index] - 1
                inventory[index] = 0
            yield (verb, color)

    def __len__(self):
        return self.limit


class ModifiedRequestGenerator:
    """
    Produces requests that allow the existence of an optimal policy
    which performs better than a greedy policy.

    Makes red items 4 times as likely as other colors.
    Color probability of restore requests does not depend
    on the number of items of that color in the inventory
    (within the bounds of still generating valid requests).
    """
    def __init__(self, pd: ProblemDescription, limit: int = 1000000):
        self.pd = pd
        self.limit = limit
        self.seed = random.randrange(int(1e100))

    def __iter__(self):
        return self.iterate()

    def iterate(self):
        rand_gen = random.Random(self.seed)
        pd = self.pd
        inventory = [0] * pd.number_of_inventory_slots
        for _ in range(self.limit):
            color_probabilities = [1] * pd.number_of_colors
            color_probabilities[1] *= 4
            index = rand_gen.randrange(pd.number_of_inventory_slots)
            verb = 0 if inventory[index] == 0 else 1
            if verb == 0:
                color = rand_gen.choices(range(pd.number_of_colors), color_probabilities)[0]
                inventory[index] = color + 1
            else:
                number_of_inventory_slots_by_color = [
                    sum((1 for item in inventory if item == color + 1))
                    for color in range(pd.number_of_colors)
                ]
                color_probabilities = [
                    color_probabilities[color] if number_of_inventory_slots_by_color[color] > 0 else 0
                    for color in range(pd.number_of_colors)
                ]
                color = rand_gen.choices(range(pd.number_of_colors), color_probabilities)[0]
                for index in range(len(inventory)):
                    if inventory[index] == color + 1:
                        inventory[index] = 0
                        break
            yield (verb, color)

    def __len__(self):
        return self.limit


class GeneratedData:
    """
    Wraps a RequestGenerator and behaves like the Data class,
    so it can be used in place of a Data instance.
    """
    def __init__(self, pd: ProblemDescription, limit: int = 100000):
        self.pd = pd
        self.limit = limit
        self.requests = RequestGenerator(pd, limit)
        self.filepath = 'generated data'

    def print_statistics(self):
        """
        prints results of statistical analysis of this data
        """
        print(self.filepath)
        print('-----------------------------------------------------------')
        stats = Statistics(self)
        stats.do_basic_analysis()
        stats.print_basic_counts()


class PatternAnalysis:
    """
    Used to find sequential request patterns in data
    by comparing it with other data without these patterns.

    Can find verb patterns, color patterns, or both.
    Sequences can only contain gaps (placeholders).

    Use method compare() with GeneratedData as a reference.
    """
    def __init__(self, pd: ProblemDescription, data: typing.Union[Data, GeneratedData],
                 pattern_length: int, use_verb: bool = True, use_color: bool = True, with_placeholder: bool = True):
        self.pd = pd
        self.data = data
        self.pattern_length = pattern_length
        self.use_verb = use_verb
        self.use_color = use_color
        self.with_placeholder = with_placeholder
        if not (0 < pattern_length < 7):
            raise ValueError
        if not (use_verb or use_color):
            raise ValueError
        self.pattern_counts: typing.Optional[collections.OrderedDict] = None
        self.requests_count = len(data.requests)

    def get_pattern_count(self):
        pd = self.pd
        variables_per_first_or_last_request = (
                (pd.number_of_verbs if self.use_verb else 1) * (pd.number_of_colors if self.use_color else 1)
        )
        if self.with_placeholder:
            variables_per_request = variables_per_first_or_last_request + 1
        else:
            variables_per_request = variables_per_first_or_last_request
        count = 1
        for i in range(1, self.pattern_length + 1):
            if i == 1 or i == self.pattern_length:
                count *= variables_per_first_or_last_request
            else:
                count *= variables_per_request
        return count

    def get_iterator_at(self, index):
        first_or_last = index == 0 or index == self.pattern_length - 1
        with_placeholder = (not first_or_last) and self.with_placeholder
        if self.use_verb:
            verbs = list(range(self.pd.number_of_verbs))
        else:
            verbs = [self.pd.number_of_verbs]
        if self.use_color:
            colors = list(range(self.pd.number_of_colors))
        else:
            colors = [self.pd.number_of_colors]
        for v in verbs:
            for c in colors:
                yield (v, c)
        if with_placeholder:
            yield (self.pd.number_of_verbs, self.pd.number_of_colors)

    def get_iterator_from(self, index):
        if index == self.pattern_length - 1:
            for request in self.get_iterator_at(index):
                yield [request]
        else:
            for request in self.get_iterator_at(index):
                for requests in self.get_iterator_from(index + 1):
                    concatenated_requests = [request]
                    concatenated_requests.extend(requests)
                    yield concatenated_requests

    def request_to_string(self, request) -> str:
        verb_symbols = [verb[0] for verb in self.pd.verbs]
        verb_symbols.append('?')
        color_symbols = [color[0] for color in self.pd.colors]
        color_symbols.append('?')
        (v, c) = request
        return verb_symbols[v] + color_symbols[c]

    def pattern_to_string(self, pattern) -> str:
        return '-'.join([self.request_to_string(request) for request in pattern])

    def pattern_matches(self, pattern, request_history) -> bool:
        assert self.pattern_length == len(pattern)
        assert self.pattern_length == len(request_history)
        match = True
        for i in range(self.pattern_length):
            mask = pattern[i]
            request = request_history[i]
            (mask_verb, mask_color) = mask
            (request_verb, request_color) = request
            if mask_verb < self.pd.number_of_verbs and mask_verb != request_verb:
                match = False
            elif mask_color < self.pd.number_of_colors and mask_color != request_color:
                match = False
            if not match:
                break
        return match

    def analyze(self):
        patterns = []
        pattern_counts = []
        for (pattern_index, pattern) in enumerate(self.get_iterator_from(0)):
            patterns.append(pattern)
            pattern_counts.append(0)
        request_history = collections.deque([], maxlen=self.pattern_length)
        for request in self.data.requests:
            request_history.append(request)
            if len(request_history) == self.pattern_length:
                for (pattern_index, pattern) in enumerate(patterns):
                    if self.pattern_matches(pattern, request_history):
                        pattern_counts[pattern_index] += 1
        self.pattern_counts = collections.OrderedDict([
            (self.pattern_to_string(patterns[i]), pattern_counts[i])
            for i in range(len(patterns))
        ])

    def compare(self, ref_pa: 'PatternAnalysis', min_expected=50, prob_ratio_threshold=2):
        """
        Analyzes how data from different sources differs.
        """
        assert self.pattern_length == ref_pa.pattern_length
        assert self.pd == ref_pa.pd
        if self.pattern_counts is None:
            self.analyze()
        if ref_pa.pattern_counts is None:
            ref_pa.analyze()
        assert len(self.pattern_counts) == len(ref_pa.pattern_counts)
        if self.use_verb and self.use_color:
            pattern_type = 'verb+color'
        elif self.use_verb:
            pattern_type = 'verb only'
        else:
            pattern_type = 'color only'
        print('differences in length {} pattern frequencies (by factor {} or more) for {}:'.format(
            self.pattern_length, prob_ratio_threshold, pattern_type))
        for pattern, count in self.pattern_counts.items():
            ref_count = ref_pa.pattern_counts[pattern]
            expected = round(self.requests_count * ref_count / ref_pa.requests_count)
            if expected < min_expected:
                continue
            ratio = count / self.requests_count
            ref_ratio = ref_count / ref_pa.requests_count
            if ratio == 0.0 or ratio/ref_ratio >= prob_ratio_threshold or ref_ratio/ratio >= prob_ratio_threshold:
                print('{pattern}: {ratio:.7f} vs {ref_ratio:.7f} expected (#: {count} vs {expected} expected)'.format(
                    pattern=pattern, ratio=ratio, ref_ratio=ref_ratio, count=count, expected=expected
                ))
