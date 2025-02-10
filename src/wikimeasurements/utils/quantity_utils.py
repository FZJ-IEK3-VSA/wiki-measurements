import re
from re import Pattern
import json
import math
import locale
import warnings
import itertools
import unicodedata
from decimal import *
from typing import Union
from fractions import Fraction
import lupa
import numpy as np
import numexpr as ne
from text_to_num import text2num
from wikimeasurements.utils.type_aliases import Labels, UnitFreqs


# Load MediaWiki modules written in Lua
lua = lupa.LuaRuntime()
CONVERT_NUMERIC = lua.require("./src/wikimeasurements/mediawiki_modules/mediawiki-extensions-Scribunto/includes/engines/LuaCommon/lualib/ConvertNumeric")

# Numbers occur in many different forms:
#   * cardinals (e.g., "27"),
#   * ordinals (e.g., "27." or "27th"),
#   * fractions (e.g., "1/27"),
#   * numbers with suffixes (e.g., "27-year"),
#   * the spelled out versions (e.g., "twenty-seven"),
#   * and further writing styles
#     (e.g., different thousands separators,  e.g., "1'234" vs. "1234",
#            or powers of ten, e.g., "2.7×10^6" or "2.6M" or "2.6 million",
#            keep trailing zeros, trim zeros, and more)

def get_number_patterns() -> tuple[Pattern]:
    """Create REGEX patterns for matching numbers"""
    # TODO: include ∓±, ~, ∼ etc. which requires an adjustment of str2num to handle these chars
    boundary = r"(^|(?<=[\s(]))"
    sign = r"(?:[++\-−‐‑‒–—―] ?)?"
    mantissa = r"(?:\d+(?:[., ']?\d{3})*(?:[,.]?\d+)?)"
    power_of_n = r"(?:(?:[eE]|(?:(?: ?[x\W])? ?\d*[\^ ]?))? ?[+\-]? ?(?:\d+[., ']?)+)?"
    num_regex = sign + mantissa + power_of_n
    fraction_regex = num_regex + r"(?:[\/\⁄]" + num_regex + r")*"  # allow fractions
    number_pattern = re.compile(boundary + fraction_regex)
    split_digits_and_words = re.compile(r"([^\W\d_]+)|(" + fraction_regex + r")")
    ordinal_pattern = re.compile(r"\d(st|nd|rd|th)\b")

    return number_pattern, split_digits_and_words, ordinal_pattern

NUMBER_PATTERN, SPLIT_DIGITS_AND_WORDS, _ = get_number_patterns()

def get_plural_form(num_word: str) -> str:
    """Get plural form of a number word."""
    if num_word.endswith("s"):
        return num_word
    elif num_word.endswith("y"):
        return num_word[:-1] + "ies"
    elif num_word.endswith("x"):
        return num_word + "es"
    else:
        return num_word + "s"

with open("src/wikimeasurements/static_resources/basic_number_words.json", "r", encoding="utf-8") as f:
    BASIC_NUMBER_WORDS = json.load(f)["number_words"]

BASIC_NUMBER_WORDS["cardinals_plural"] = {get_plural_form(k): v for k, v in BASIC_NUMBER_WORDS["cardinals"].items()}
BASIC_NUMBER_WORDS["ordinals_plural"] = {get_plural_form(k): v for k, v in BASIC_NUMBER_WORDS["ordinals"].items()}


def get_basic_number_words(numeral_type: str = "all") -> list[str]:
    """Get alphabtic numbers from Mediawiki Module "ConvertNumeric" """

    number_words = []
    if numeral_type in ["ordinals", "ordinals_minus_denominator_intersection", "all"]:
        ordinals = list(BASIC_NUMBER_WORDS["ordinals"].keys())
        if numeral_type == "ordinals_minus_denominator_intersection":
            # Some ordinals like 'third' are also denominators as in 'one-third'.
            # Remove the intersecting terms.
            ordinals = [o for o in ordinals if o not in BASIC_NUMBER_WORDS["denominators"].keys()]
            ordinals += [get_plural_form(o) for o in ordinals]
        else:
            ordinals += list(BASIC_NUMBER_WORDS["ordinals_plural"].keys())
                
        number_words += ordinals

    if numeral_type in ["denominators", "all"]:
        number_words += list(BASIC_NUMBER_WORDS["denominators"].keys())

    if numeral_type in ["cardinals", "all"]:                
        number_words += list(BASIC_NUMBER_WORDS["cardinals"].keys()) + list(BASIC_NUMBER_WORDS["cardinals_plural"].keys())

    if numeral_type in ["magnitudes", "all"]:
        number_words += list(BASIC_NUMBER_WORDS["large_orders_of_magnitude"].keys())
    
    number_words_lowered = [term.lower() for term in list(set(number_words))]

    return number_words_lowered


def get_colloquial_number_words() -> list[str]:
    """Various quantity terms from Hanauer et al.
    "Complexities, Variations, and Errors of Numbering within Clinical Notes:
    The Potential Impact on Information Extraction and Cohort-Identification",
    2019, https://doi.org/10.1186/s12911-019-0784-1.
    """

    with open("src/wikimeasurements/static_resources/colloquial_number_words.json", "r", encoding="utf-8") as f: 
        data = json.load(f)

    colloquial_number_words = []
    for table in data["number_words"].values():
        colloquial_number_words += table["content"]

    colloquial_number_words.remove("less")

    return colloquial_number_words


def get_physical_constants_from_wikidata() -> list[str]:
    """Various terms of physical quantities from Wikidata"""

    with open("src/wikimeasurements/static_resources/physical_constants.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["physical_constants"]


def get_number_words_quantifier_and_physical_constants() -> tuple[list[str], list[str]]:

    basic_num_words = get_basic_number_words()
    colloquial_num_words = get_colloquial_number_words()
    physical_constants = get_physical_constants_from_wikidata()    
    custom_terms = ["few ", "room temperature", "golden ratio", "Pi "]
    all_terms = set(basic_num_words + colloquial_num_words + physical_constants + custom_terms)

    # Filter out terms containing numeric chars, since those are filtered out separately.
    num_chars = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "²", "³"]
    all_alpha_only_terms = [term for term in all_terms if not any(num in term for num in num_chars)]

    return all_alpha_only_terms, basic_num_words


def num_word_to_num(clean_string: str) -> Union[float, int]:
    """Convert a number word like "five", "fith", "million", etc. into an integer or float.
    Note that a billion has two distinct definitions, 10^9 (short scale) and 10^12 (long scale), 
    of which the former one is used in English languages.
    """
    number = BASIC_NUMBER_WORDS["cardinals"].get(clean_string)
    if number == None:
        number = BASIC_NUMBER_WORDS["ordinals"].get(clean_string)
        if number == None:
            number = BASIC_NUMBER_WORDS["cardinals_plural"].get(clean_string)
            if number == None:
                number = BASIC_NUMBER_WORDS["ordinals_plural"].get(clean_string)
                if number == None:
                    number = BASIC_NUMBER_WORDS["denominators"].get(clean_string)
                    if number != None:
                        number = float(number)
                    else:
                        number = BASIC_NUMBER_WORDS["large_orders_of_magnitude"].get(clean_string)
                        if number != None:
                            number = 10 ** (3 * number)
    return number


def cast_str_as_int(num_str: str) -> int:
    """Cast string as integer."""
    return locale.atoi(num_str)


def cast_str_as_float(num_str: str) -> float:
    """Cast string as float."""
    return locale.atof(num_str)


def cast_str_as_fraction_sum(num_str: str) -> float:
    """Cast string as sum of fractions."""
    # Replace "//" with "/" and delete whitespace around "/"
    fract_string = re.sub(r"(?<=\d)(\s*/{1,2}\s*)(?=\d)", "/", num_str)
    fract_string = re.sub(r"(?<=\d)(\s+/\s+)(?=\d)", "/", num_str)

    # For seperation of values at whitespace convert
    # '-2-1/4' to '-2 -1/4' and '-2 + 1/4' to '-2 +1/4', etc.
    fract_string = re.sub(r"(?<=\d)(\s*-\s*)(?=\d)", " -", fract_string)
    fract_string = re.sub(r"(?<=\d)(\s*\+\s*)(?=\d)", " +", fract_string)

    return float(sum(Fraction(num) for num in fract_string.split()))


def cast_str_as_number_words(num_str: str) -> Union[float, int]:
    """Cast string as special number words not coverd
    by below method like ordinals or plurals
    (e.g., 'fifth' and 'fives').
    """
    parsed_num = num_word_to_num(num_str)
    if parsed_num is None:
        raise ValueError
    else:
        return parsed_num


def cast_str_as_math_expr(num_str: str) -> float:
    """Cast string as mathematical expression."""
    # First, change for example '7 10^2' to '7*10^2'
    math_string = re.sub(r"(?<=\d)(\s+)(?=10\^\d)", "*", num_str)
    math_string = math_string.replace("^", "**").replace("x", "*").replace("×", "*")

    with warnings.catch_warnings():
        # Surpress warnings like "SyntaxWarning: 'int' object is not callable;
        # perhaps you missed a comma?" for strings like "3 (number)", which
        # should just return None but not a warning.
        warnings.simplefilter("ignore", SyntaxWarning)
        np_array = ne.evaluate(math_string)

    return float(np_array)


def cast_str_as_digits_and_number_words(num_str: str) -> Union[float, int]:
    """Cast string as a mix of digits and number words
    (e.g., five thousand or 1.2 million).
    """
    additive = (
        num_str.replace(" a ", " 1 ")
        .replace(" plus ", " and ")
        .replace(", ", " and ")
        .split(" and ")
    )
    total_sum = 0
    for num_str in additive:
        digit_word_groups = SPLIT_DIGITS_AND_WORDS.findall(num_str)
        product = 1
        for (num_words, num_digits) in digit_word_groups:
            if num_words == "":
                # Only digits
                parsed_num = str2num(num_digits, consider_num_words=False)
            else:
                # Number words or other (text2num parses, e.g., "one million" but
                # not "million", hence we first check for single number words)
                parsed_num = num_word_to_num(num_words)

                if parsed_num is None:
                    # We use text2num, since word2number is not verbose enough.
                    # For example, word_to_num("1.2 million") yielded 1000000
                    # and word_to_num("two million, one") yielded 3.
                    parsed_num = text2num(num_words, lang="en", relaxed=False)

            product *= parsed_num

        total_sum += product

    return total_sum


def str2num(string: str, consider_num_words: bool = True) -> Union[float, int]:
    """Converts e.g. "12345.0" to float and 12345 to int.
    Locale.atoi and .atof are used in order to cope with
    language specific writing of numbers, e.g., commas as
    thousands delimiters for US English. In order to change
    language add:
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

    Limitations:
        Roman numerals (e.g., 'XIII') are not supported.
        Multidim. values (e.g., '10x50x100') are not supported.

    A note on the order:
    The methods for casting strings as fractions, mathematical
    expression and mixture of digits and num words are slower
    than for integers, floats and number words. Additionally,
    the respective kinds of number strings occur probably much
    less frequent. Therefore, they are placed last.
    """

    # Setting localization could also be done once
    # importing this module, however, when using Spacy
    # in combination with benepar, it will change the
    # localization setting. Thus, to be certain about
    # the localization setting, it is defined here.
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

    # Trim whitespace and normalize
    string = string.strip().replace("−", "-").replace("⁄", "/")

    # Convert ordinals like '30th' to '30'
    string = re.sub(r"(?<=\d)(st|nd|rd|th)$", "", string)

    # Handle special unicode numbers (e.g., 1⁄4, ¼ or ¹/₇₉₈)
    # Normalize string such that '¼' is turned to "1⁄4"
    clean_string = ""
    already_normalized = False
    for char in string:
        last_already_normalized = already_normalized
        already_normalized = unicodedata.is_normalized("NFKC", char)
        # Whitespace at boundary (e.g., '10²³' to '10 23' not '1023')
        boundary = "" if last_already_normalized == already_normalized else " "
        normalized_char = (
            char if already_normalized else unicodedata.normalize("NFKC", char)
        )
        clean_string += boundary + normalized_char

    # Trim whitespace
    clean_string = clean_string.strip()

    try:
        number = cast_str_as_int(clean_string)
    except:
        pass
    else:
        return number

    try:
        number = cast_str_as_float(clean_string)
    except:
        pass
    else:
        return number

    if consider_num_words:
        try:
            number = cast_str_as_number_words(clean_string)
        except:
            pass
        else:
            return number

    try:
        number = cast_str_as_fraction_sum(clean_string)
    except:
        pass
    else:
        return number

    if consider_num_words:
        try:
            number = cast_str_as_digits_and_number_words(clean_string)
        except:
            pass
        else:
            return number

    try:
        number = cast_str_as_math_expr(clean_string)
    except:
        return None  # Fail silently
    else:
        return number


def get_fraction_str(number, fraction_line="/", thousands_sep=""):
    """Convert a numeric value into fraction string representation.
        For example, given 0.24 the string '6/25' is returned.

    :param number: input number
    :type number: float or int
    :param fraction_line: fraction line, defaults to "/"
    :type fraction_line: str, optional
    :param thousands_sep: thousands sperator, defaults to ""
    :type thousands_sep: str, optional
    :return: string representation of fraction
    :rtype: str
    """
    fraction = Fraction(number).limit_denominator(max_denominator=1000)
    numerator_str = f"{fraction.numerator:,}".replace(",", thousands_sep)
    denominator_str = f"{fraction.denominator:,}".replace(",", thousands_sep)
    fraction_str = numerator_str + fraction_line + denominator_str

    return fraction_str


def num2str(
    num,
    base="×10^",
    exp=0,
    spell_magn=False,
    thousands_sep="",
    prec=2,
    pad_exp=0,
    show_plus=False,
    fraction=False,
    fraction_sign="/",
    fraction_exp=False,
):

    # Some options validity checks
    # ...

    # Adapt precision, e.g., to not yield '0x10^+1' for '1' given the exponent 1.
    prec = prec + exp

    # Get string representation of the mantissa
    mantissa = num * 10 ** (-exp)

    if prec < 0:
        # round mantissa
        mantissa = round(mantissa, prec)
        prec = 0  # set precision for decimals to 0

    if fraction:
        mantissa_str = get_fraction_str(mantissa, fraction_sign, thousands_sep)
    else:
        mantissa_str = f"{mantissa:,.{prec}f}".replace(",", thousands_sep)

    # Get string representation of the order of magnitude
    if exp == 0:
        magn_str = ""  # 123×10^-0 etc. is not common
    else:
        if spell_magn and (exp > 0) and (exp % 3 == 0):
            magnitude_words = list(BASIC_NUMBER_WORDS["large_orders_of_magnitude"].keys())
            magn_word = magnitude_words[(exp // 3) - 1]
            magn_str = " " + magn_word
        else:
            sign = ("+" if show_plus else "") if exp > 0 else "-"
            if fraction_exp:
                exp_str = get_fraction_str(abs(exp), fraction_sign, thousands_sep)
            else:
                exp_str = f"{abs(exp):0>{pad_exp}}"

            magn_str = base + sign + exp_str

    # Combine all
    num_str = mantissa_str + magn_str

    return num_str


def get_number_spellings(num: str, numerator: str, denominator: str):
    """Spell out a number using MediaWiki's ConvertNumeric module
        (https://en.wikipedia.org/wiki/Module:ConvertNumeric)

    :param num: number (None if no whole number before a fraction)
    :type num: str or None
    :param numerator: numerator of fraction (None if no fraction)
    :type numerator: str or None
    :param denominator: denominator of fraction (None if no fraction)
    :type denominator: str or None
    :return: unique English spellings of the given number
    :rtype: list
    """

    # Relavant option per argument of spell_number() in ConvertNumeric module
    capitalize = [False]
    use_and = [True, False]
    hyphenate = [True, False]
    ordinal = [True, False]
    plural = [True, False]
    links = [None]
    negative_word = [None]
    round = [None]  # [None, "up", "down"]
    zero = [None]  # [None, "nil", "null"]
    use_one = [True, False]

    # For explainational comments check source code of ConvertNumeric module
    spelling_options = [
        [num],
        [numerator],
        [denominator],
        capitalize,
        use_and,
        hyphenate,
        ordinal,
        plural,
        links,
        negative_word,
        round,
        zero,
        use_one,
    ]
    permutation = list(itertools.product(*spelling_options))
    spellings = [CONVERT_NUMERIC[0].spell_number(*variant) for variant in permutation]
    unique_spellings = list(set(spellings))

    return unique_spellings


def get_digit_notations(number: str, threshold=0.03):

    # TODO: Check https://en.wikipedia.org/w/index.php?title=Orders_of_magnitude_(numbers)&action=edit
    #       Is 1000<sup>−10</sup> and 1.24{{e|−68}} parsed correctly?

    n = float(number)

    # Get base notation options
    # I expect "×" to be normalized to "x", do not expect dot
    ten_notations = [
        ["x", " x ", " ", ""],
        ["10"],  # base is always ten
        ["^", ""],
    ]
    e_notations = ["e", "E"]
    base_options = ["".join(b) for b in itertools.product(*ten_notations)] + e_notations

    # Get exponent options
    magnitude = 0 if n == 0 else math.floor(math.log10(abs(n)))
    exp_range = 2
    min_exp = 3 * math.floor((magnitude - exp_range) / 3)
    max_exp = 3 * math.ceil((magnitude + exp_range) / 3)
    exp_options = list(range(min_exp, max_exp + 1))

    spell_magn_options = [True, False]

    # Thousands seperator options
    # I do not expect thin space (" ") or underscore ("_")
    sep_options = ["", ",", " ", "'"]

    # Precision options
    n_dec = Decimal(number).as_tuple()
    decimal_places = -n_dec.exponent
    integer_places = len(n_dec.digits) - decimal_places
    all_prec_options = list(range(-integer_places, decimal_places + 1))
    all_prec_options.reverse()
    if n == 0:
        valid_prec_options = all_prec_options
    else:
        valid_prec_options = []
        for prec in all_prec_options:
            deviation = MAPE(n, round(n, prec))
            if deviation > threshold:
                break
            else:
                valid_prec_options.append(prec)
    
    # # Leave the first three places untouched to limit deviation (1000 / 1009 > 0.99)
    # min_prec = min(0, -integer_places + 3)
    # max_prec = decimal_places

    # # Precision shall not be higher than precision of input number
    # prec_options = list(range(min_prec, max_prec + 1))

    pad_exp_options = [0, 1]

    # Sign options
    show_plus_options = [True, False]

    scientific_notation_options = [
        base_options,
        exp_options,
        spell_magn_options,
        sep_options,
        valid_prec_options,
        pad_exp_options,
        show_plus_options,
    ]

    permutation = list(itertools.product(*scientific_notation_options))
    notations = [num2str(n, *variant) for variant in permutation]

    return list(set(notations))


def MAPE(y_true: float, y_pred: float) -> float:
    """Calculate the Mean Absolute Percentage Error (MAPE)
    for a series of length 1. This implementation is
    similar to the implementation in scikit-learn.
    Epsilon is used in order to not devide by zero."""
    return abs(y_pred - y_true) / max(abs(y_true), np.finfo(np.float64).eps)


def is_small_int(value: Decimal, threshold: int = 10) -> bool:
    """Determine if a value is an integer below a
    certain threshold."""
    (numerator, denominator) = value.as_integer_ratio()
    if denominator == 1 and abs(numerator) < threshold:
        return True
    else:
        return False


celsius_to_kelvin = lambda t: t + 273.15
kelvin_to_celsius = lambda t: t - 273.15
celsius_to_fahrenheit = lambda t: (t * (9 / 5)) + 32
fahrenheit_to_celsius = lambda t: (t - 32) * 5 / 9


def convert_temperature(
    initial_value_str: str, initial_unit: str
) -> list[dict[str, Union[Decimal, str]]]:
    """Converting between Kelvin, degree Celsius and degree Fahrenheit
    is different from converting most other units, as it requires
    addition and subraction. Therefore, we cannot use predefined
    conversion factors, but calculate the converted values according
    to the different temperature scales individually for each value.
    """

    temperature_units = {
        "celsius": "Q25267",
        "kelvin": "Q11579",
        "fahrenheit": "Q42289",
    }

    initial_unit = initial_unit.removeprefix("wd:")

    # Get values according to different temperature scales
    if initial_unit == temperature_units["celsius"]:
        celsius = float(initial_value_str)
        kelvin = celsius_to_kelvin(celsius)
        fahrenheit = celsius_to_fahrenheit(celsius)

    elif initial_unit == temperature_units["kelvin"]:
        kelvin = float(initial_value_str)
        celsius = kelvin_to_celsius(kelvin)
        fahrenheit = celsius_to_fahrenheit(celsius)

    elif initial_unit == temperature_units["fahrenheit"]:
        fahrenheit = float(initial_value_str)
        celsius = fahrenheit_to_celsius(fahrenheit)
        kelvin = celsius_to_kelvin(celsius)

    else:
        raise ValueError

    conversion_factor = None
    considered_units = [
        (
            {
                "converted_temp_value": Decimal(celsius),
                "to_unit": temperature_units["celsius"],
            },
            conversion_factor,
        ),
        (
            {
                "converted_temp_value": Decimal(kelvin),
                "to_unit": temperature_units["kelvin"],
            },
            conversion_factor,
        ),
        (
            {
                "converted_temp_value": Decimal(fahrenheit),
                "to_unit": temperature_units["fahrenheit"],
            },
            conversion_factor,
        ),
    ]

    return considered_units


def get_frequent_units(
    LABELS: Labels,
    unit_freqs: UnitFreqs,
    mode: str = "top_x_of_not_zero",
    top_x_percent: int = 50,
) -> list[str]:
    """Get frequently used units."""

    not_zero = [q for q, count in unit_freqs["frequencies_of_units_with_label"] if count > 0]

    if mode == "not_zero":
        frequent_unit_uris = not_zero
    else:
        if mode == "top_x_of_not_zero":
            considered_freqs = not_zero
        elif mode == "top_x":
            considered_freqs = [q for q, count in unit_freqs["frequencies_of_units_with_label"]]
        else:
            raise ValueError

        frequent_unit_uris = [q for q in considered_freqs[ : int(len(considered_freqs) * top_x_percent / 100)]]

    # Get aliases
    entity_base_url = "http://www.wikidata.org/entity/"
    frequent_unit_aliases = [
        LABELS["units"].get(entity_base_url + Q)
        for Q in frequent_unit_uris if LABELS["units"].get(entity_base_url + Q) is not None
    ]

    frequent_unit_aliases = list(set(itertools.chain(*frequent_unit_aliases)))

    # Remove units without label and those which are, in fact, numbers and not units
    if None in frequent_unit_aliases:
        frequent_unit_aliases.remove(None)

    frequent_units = [alias for alias in frequent_unit_aliases if str2num(alias) is None]    
    frequent_units.sort()

    return frequent_units
