import pytest
from wikimeasurements.utils.quantity_utils import str2num


def test_str2num():
    """Test if str2num correctly parses strings to numbers."""

    test_strings = [
        ("Ce n'est pas un nombre", None),
        ("123", 123),
        ("-123", -123),
        ("123.456", 123.456),
        ("-123.456", -123.456),
        ("123,456", 123_456),
        ("9 3/4", 9.75),
        ("9 + 3/4", 9.75),
        ("9 -3/4", 8.25),
        ("9 - 3/4", 8.25),
        ("3*10^6", 3 * 10 ** 6),
        ("1 million", 1_000_000),
        ("five thousand", 5_000),
        ("thousand", 1_000),
        ("billion", 10 ** 9),
        ("one and a half", 1.5),
        ("gazillion", None),
    ]

    for num_str, num_solution in test_strings:
        num_result = str2num(num_str)
        assert num_result == num_solution


if __name__ == "__main__":
    test_str2num()
