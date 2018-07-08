import operator
import re
from src.tools.osu.utils import is_float

OPERATORS = {'~': operator.contains, '=': operator.eq, '>': operator.gt, '<': operator.lt, '>=': operator.ge,
             '<=': operator.le}
OP_KEYS = list(OPERATORS.keys())
OP_KEYS.sort(key=len, reverse=True)
OPERATOR_REGEX = r'''({0}|\*|\n)'''.format('|'.join(OP_KEYS))


def filter_item(item, filters):
    for f in filters:
        if not f.check(item):
            return False
    return True


def filter_items(data: list, filters) -> list:
    return [item for item in data if filter_item(item, filters)]


class Filter:
    __slots__ = ['field', 'operator', 'value']

    def __init__(self, field, operator, value):
        self.field = field
        self.operator = operator
        if value.isdigit() or is_float(value):
            self.value = eval(value)
        else:
            self.value = str(value).replace('\'', '').replace('"', '')

    def check(self, data):
        return self.operator(getattr(data, self.field), self.value)

    def __str__(self) -> str:
        op_str = '?'
        for op in OPERATORS.keys():
            if OPERATORS[op] == self.operator: op_str = op
        return 'Filter: {0}{1}{2}'.format(self.field, op_str, self.value)


def parse_filters(text):
    # Split string by ' ' but keep spaces in strings enclosed in ""
    raw_filters = re.split(''' (?=(?:[^']|'[^']*')*$)''', text)
    ret = []
    for raw_filter in raw_filters:
        seperated = list(filter(None, re.split(OPERATOR_REGEX, raw_filter)))
        if len(seperated) != 3: continue
        ret.append(Filter(seperated[0], OPERATORS[seperated[1]], seperated[2]))
    return ret
