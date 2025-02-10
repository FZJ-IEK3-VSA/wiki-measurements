from collections import defaultdict
from typing import Union


# Type aliases
Offset = tuple[int, int]
Annotations = Union[dict[str, list[Offset]], defaultdict[list[Offset]]]
Facts = dict[str, list[dict]]
Labels = dict[str, dict[str, list[str]]]
UnitConvData = dict[str, list[dict[str, str]]]
UnitFreqs = dict[str, Union[list[str], list[list[str, int]]]]
