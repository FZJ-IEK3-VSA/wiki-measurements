import sys
import logging
import traceback
from datetime import datetime
from dateparser.date import DateDataParser
from dateparser_data.settings import default_parsers


# Get date parser which does not construct an absolute date
# such as datetime.datetime(2022, 9, 13, 0, 0) for relative
# dates like "2 weeks" or incomplete datas like "September 13".
parsers = [parser for parser in default_parsers if parser != "relative-time"]
date_parser = DateDataParser(
    languages=["en"],
    settings={
        "PARSERS": parsers,
        "PREFER_DATES_FROM": "past",
        "REQUIRE_PARTS": ["year"],
    },
)

def print_exception_traceback(e, logger=None):
    """Print exception traceback"""
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    message = "".join(lines)
    if logger == None:
        print(message)
    else:
        logger.log(logging.ERROR, message)


def compare_datetimes(
    known_date: datetime, candi_date_str: str, known_prec: str = "day"
) -> bool:
    """Returns true if dates match according to precision else false."""

    # We use get_date_data() instead of using dateparser.parse() directly,
    # in order to get information about the precision for incomplete dates.
    parser_result = date_parser.get_date_data(candi_date_str)
    candi_date = parser_result.date_obj

    if candi_date is None:
        # Nothing to compare
        return False

    prec_map = {
        "year": 0,
        "month": 1,
        "day": 2,
        "hour": 3,
        "minute": 4,
        "second": 5,
    }

    # Default to day
    default_prec = "day"

    # Get precision of known date
    known_date_prec = prec_map.get(known_prec)
    if known_date_prec is None:
        known_date_prec = prec_map[default_prec]

    # Get precision of candidate date
    candi_date_prec = prec_map.get(parser_result.period)
    if candi_date_prec is None:
        candi_date_prec = prec_map[default_prec]

    # Take the lowest of both precisions
    prec = min(known_date_prec, candi_date_prec)

    # Check if dates match according to precision
    # Match on years
    if candi_date.year != known_date.year:
        return False
    elif prec == prec_map["year"]:
        return True  # dates match!

    # Match on months
    if (prec >= prec_map["month"]) and (candi_date.month != known_date.month):
        return False
    elif prec == prec_map["month"]:
        return True  # dates match!

    # Match on days
    if (prec >= prec_map["day"]) and (candi_date.day != known_date.day):
        return False
    elif prec == prec_map["day"]:
        return True  # dates match!

    # Match on hours
    if (prec >= prec_map["hour"]) and (candi_date.hour != known_date.hour):
        return False
    elif prec == prec_map["hour"]:
        return True  # dates match!

    # Match on minutes
    if (prec >= prec_map["minute"]) and (candi_date.minute != known_date.minute):
        return False
    elif prec == prec_map["minute"]:
        return True  # dates match!

    # Match on seconds
    if (prec >= prec_map["second"]) and (candi_date.second != known_date.second):
        return False
    else:
        return True  # dates match!


def init_logger(filename, log_level):
    """Initialize a logger."""

    logger = logging.getLogger(filename)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(filename, encoding="utf-8"),
        ],
    )

    # Hide logging of some libraries
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger.setLevel(log_level)
    logger.info("Start logging...")

    return logger
