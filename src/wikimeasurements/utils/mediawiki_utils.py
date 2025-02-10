import re
import requests
import unicodedata
import mwparserfromhell
from wikiextractor.extract import Extractor
import lupa


# Load MeadiaWikis convert template, which is written in Lua.
lua = lupa.LuaRuntime()
lua_fun_code = """
    function(x)
        -- Load convert module
        local wiki_convert = require("./src/wikimeasurements/mediawiki_modules/mediawiki-extensions-Scribunto/includes/engines/LuaCommon/lualib/convert")

        -- Process convert template
        result = wiki_convert.convert(x)
        return result
    end
"""
cvt = lua.eval(lua_fun_code)

ex = Extractor(None, None, None, None, None)

# Pre-compile regex patterns
lost_insertion = r"(, ,|- -|– –|— —)"
lost_parenthesis_content = r"\((\s?[,;]?s?)*\)"
partially_lost_parenthesis_content = r"(?<=\()[,; ]+(?=\w)"
regex = (
    r"("
    + lost_insertion
    + r"|"
    + lost_parenthesis_content
    + r"|"
    + partially_lost_parenthesis_content
    + r")"
)
LOST_INSERTION_PATTERN = re.compile(regex)
MULTI_WHITESPACE_PATTERN = re.compile(r" {2,}")

def get_wikipedia_statistics(lang: str = "en", print_stats: bool = False):
    """Get current statistics of Wikipedia. Per default
    the stats are queried for the English Wikipedia.
    """

    try:
        url = f"https://{lang}.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=statistics&format=json"
        result = requests.get(url).json()
        wiki_stats = result["query"]["statistics"]
        print(f"Queried current statistics of {lang} Wikipedia")
    except:
        # If no internet connection available,
        # take stats accessed on 2022-08-18.
        if lang == "en":
            wiki_stats = {
                "pages": 56797145,
                "articles": 6571321,
                "edits": 1112689654,
                "images": 895890,
                "users": 44364458,
                "activeusers": 114683,
                "admins": 1031,
                "jobs": 0,
                "cirrussearch-article-words": 4185916131,
                "queued-massmessages": 0,
            }
        elif lang == "simple":
            wiki_stats = {
                "pages": 716324,
                "articles": 220357,
                "edits": 8469231,
                "images": 36,
                "users": 1240079,
                "activeusers": 1093,
                "admins": 18,
                "jobs": 0,
                "cirrussearch-article-words": 48570822,
                "queued-massmessages": 0,
            }
        else:
            raise ValueError(
                f"Seems as if no internet connection is available and no offline stats are provided for selected language {lang} in get_wikipedia_statistics()."
            )

        print(
            f"Failed to query current statistics of {lang} Wikipedia. Thus, take hardcoded statistics from 2022-08-18."
        )

    if print_stats:
        print(wiki_stats)

    return wiki_stats


def clean_wiki_page(raw_page):
    """Clean page"""

    # Remove infoboxes and sidebars
    wikicode = mwparserfromhell.parse(raw_page)
    for template in wikicode.filter_templates():
        name = template.name.lower()
        if name.startswith("infobox") or name.startswith("sidebar"):
            try:
                wikicode.remove(template)
            except:
                pass

    # Alternatively, use regex for detecting infoboxes
    # infobox_regex = r"({{(?:Infobox).+?(?=}})}})"
    # infobox_pattern = re.compile(infobox_regex, flags=re.IGNORECASE | re.DOTALL)

    clean_page = str(wikicode)

    # Remove HTML comments
    clean_page = re.sub("(<!--.*?-->)", "", clean_page, flags=re.DOTALL)

    # TODO: cope with cases like
    # [[File:Susitnabridge.JPG|thumb|left|The [[Susitna River]] bridge on the [[Denali Highway]] is {{convert|1036|ft}} long.]]
    # (https://en.wikipedia.org/wiki/Alaska)

    return clean_page


def clean_wiki_template(raw_text: str) -> str:
    """Clean a raw template text for further processing and
        return clean inner template text. Cleaning involves
        removing parentheses, referenses and page links. The
        latter is especially important for not splitting a
        page link at "|" into two args.

        Examples:
            * '{{convert|123<ref>Lorem Ipsum</ref>|km|m}}'
                --> 'convert|123|km|m'
            * '{{convert|14|m|ft|adj=mid|[[Draft (hull)|draught]]}}'
                --> 'convert|14|m|ft|adj=mid|draught'

    :param raw_text: Raw template text (e.g., '{{cvt|123<ref>...</ref>|kg|lb}}')
    :type raw_text: str
    :return: Clean inner template text (e.g., 'cvt|123|kg|lb')
    :rtype: str
    """

    # Remove parentheses
    text = raw_text.lstrip("{{").rstrip("}}")

    # Remove references
    text = re.sub("(<ref.*?/ref>)", "", text, flags=re.DOTALL)

    # Remove HTML comments, convert HTML char notations to unicode
    # (e.g., "&minus;5" to "-5") and parse links in Mediawiki markup
    # (e.g., "[[Draft (hull)|draught]] to "draught"").
    # For now we use wikiextractor. BeautifulSoup could also be used:
    #   text = BeautifulSoup(text, "html.parser").text
    clean_text = ex.clean_text(text, mark_headers=False)

    return "".join(clean_text)


def create_convert_template_frame(text: str) -> dict:
    """Parse the inner text of a convert template into args and
       create a dict which imitates a MediaWiki frame object.

       The frame object holds all the args and is used by MediaWiki
       to pass args from MediaWiki PHP to Lua modules.

       Note that this method was only written for parsing the args
       of the convert template.

       Ressources:
           * [Frame object](https://www.mediawiki.org/wiki/Extension:Scribunto/Lua_reference_manual#Frame_object)

    :param text: inner text of a template (e.g., 'convert|15|km|mi')
    :type text: str
    :return: Imitation of a MediaWiki frame object
    :rtype: dict
    """

    # TODO: Maybe use parsed args from mwparserfromhell instead

    # Init Python dict for holding frame arguments
    frame_dict = {}

    # Parse arguments
    raw_args = text.split("|")[1:]

    if text.lower().startswith("cvt"):
        # {{cvt...}} is an alias for {{convert.. }}
        # where abbr is on per default.
        frame_dict["abbr"] = "on"

    arg_index = 0
    for raw_arg in raw_args:
        arg = raw_arg.split("=")[::-1]
        if len(arg) == 1:
            # Argument is unnamed parameter.
            # Use integer as index
            arg_index += 1  # convert module begins with 1 instead of 0
            key = arg_index
        else:
            # Argument is named parameter.
            # Use name as key and value as value
            key = arg[1].strip()

            # Check if key is positional key,
            # where 1 is the value, 2 is the input unit, 3 is the
            # output unit and 4 are the "significant digits after
            # the decimal dot or, if negative, exponent of ten."
            # (https://en.wikipedia.org/wiki/Template:Convert#TemplateData)
            if key in ["1", "2", "3", "4"]:
                key = int(key)

        frame_dict[key] = arg[0].strip()

    return frame_dict


def parse_mediawiki_markup(text, debug_mode=False):
    # Prefixing some char (e.g., '🍊') prevents WikiExtractor from dropping
    # the period in cases like: ".\nIn 2021, scientists reported"
    clean_text = ex.clean_text("🍊" + text, mark_headers=True)

    if len(clean_text) == 0:
        return [""]
    else:
        # Remove the random character again
        clean_text[0] = clean_text[0][1:]

        # Add linebreaks for paragraphs and headers, but not at
        # last paragraph since it is followed by a quantity.
        clean_text = [
            MULTI_WHITESPACE_PATTERN.sub(" ", LOST_INSERTION_PATTERN.sub("", t))
            + "\n\n"
            for t in clean_text[:-1]
        ] + [clean_text[-1]]

        if debug_mode:
            if text.startswith(".") and not clean_text[0].startswith("."):
                raise ValueError(
                    "There are still some cases where punctuation marks and other text is truncated at the beginning. Investigate!"
                )
                # Example: "Some text that will be dropped.<ref>[Some ref]</ref>{}\n\n==Location==\nIt is " (https://en.wikipedia.org/wiki/Vehari)

        return clean_text


def lua_convert(frame_dict: dict) -> str:
    """Invoke MediaWikis convert module which converts numbers between
        units and returns a string of the form '15 kilometres (9.3&nbsp;mi)'.
        A Python dict which imitates a MediaWiki frame object is required
        as input. The convert module is interweaved with MediaWiki and the
        MediaWiki Scribunto extension. With some changes the link to the
        former was cut. Above other changes to the convert module and the
        Scribunto extension were made in order for the convert module to
        work stand-alone.

        Ressources:
            * [Convert template docs](https://en.wikipedia.org/wiki/Template:Convert)
            * [Convert template help](https://en.wikipedia.org/wiki/Help:Convert)
            * [Frame object](https://www.mediawiki.org/wiki/Extension:Scribunto/Lua_reference_manual#Frame_object)
            * [Scribunto extension](https://github.com/wikimedia/mediawiki-extensions-Scribunto)

    :param frame_dict: Imitation of a MediaWiki frame object
    :type frame_dict: dict
    :return: String of the form '15 kilometres (9.3&nbsp;mi)'
    :rtype: str
    """

    # Init Lua table for holding frame arguments
    frame_table = lua.eval("{ }")
    frame_table.args = lua.eval("{ }")

    # Fill Lua table from Python dict
    frame_table.args = lua.table_from(frame_dict)

    # Run convert module, that is, do the actual
    # unit conversion and create a string of the
    # input and output quantity.
    result = cvt(frame_table)

    return result


def parse_convert_template(text):
    """Parse Mediawiki's {{convert|... }} and {{cvt|...}} template.
    {{cvt|...}} is an alias for {{convert|...|abbr=on}}.

    There are many ways of using the convert template, among others:
        Standard:
            {{convert|123.4|ft|m}}
        Ranges:
            {{convert|100|x|120|x|210|mm|in}}
        Multiple components:
            {{convert|5|ft|8|in}}
        Multiple components:
            {{convert|123|nmi|km mi}}
        Attributes:
            {{convert|123.4|ft|m|abbr=off|sp=us}}
        Splitting:
            {{convert|5+3/4|in|mm|0}} → 5+3/4 inches (146 mm)

    As of February 2022, the convert module is used in approx. 2% of
    all Wikpedia pages. [1]

    The convert template is described here:
      https://en.wikipedia.org/wiki/Help:Convert

    The relevant modules written in Lua are listed here:
      https://en.wikipedia.org/wiki/Module:Convert

    ---
    [1] https://en.wikipedia.org/wiki/Module:Convert
    """

    # TODO: handle nested templates like {{convert|{{Scottish council populations|ARE=S12000033}}|km2|sqmi|abbr=on}} in article on Aberdeen
    # TODO: handle nested convert templates like {{convert|{{convert|1|ly|Pm|3|disp=number}}e12|km|e12mi|abbr=off|sp=us}} in article on interstellar travel

    # "Some infoboxes have examples like {{convert|NNN|m}} (3 or more "N")."
    # Skip them.
    # Examples of templates to skip:
    #   "{{convert|VALUE|UNITS}}",
    #   "{{cvt|VALUE|UNITS}}",
    #   "{{Convert|NN|ha|acres}}",
    #   "{{convert|weight in kg|kg|lb}}"
    # Also skip templates nested within convert template? or "|{{" in text
    if (
        "convert|nnn" in text.lower()
        or "cvt|nnn" in text.lower()
        or re.search(r"\d", text) is None
    ):
        return None

    text = clean_wiki_template(text)
    frame_dict = create_convert_template_frame(text)
    results = lua_convert(frame_dict)

    if lupa.lua_type(results) == "table":
        # Lua table to dict
        results = dict(results)

        # HTML to text
        clean_results = []
        for result in results.values():

            # Alternatively use BeautifulSoup(result, "html.parser").text
            clean_result = "".join(ex.clean_text(result, mark_headers=False))

            # Normalize unicode chars (e.g., '\xa0' to ' ')
            clean_result = unicodedata.normalize("NFKD", clean_result)

            clean_results.append(clean_result)

        return clean_results
    else:
        return None
