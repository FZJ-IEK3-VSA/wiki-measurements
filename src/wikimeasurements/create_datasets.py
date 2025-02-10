import re
import json
import math
import logging
import unicodedata
from decimal import *
from tqdm import tqdm
from typing import Union
from random import choices
from argparse import ArgumentParser
from collections import defaultdict
from wikimeasurements.utils.filters import (
    filter_sentence,
    filter_value_annotations,
    filter_unit_annotations,
    filter_context_annotations,
    filter_examples_from_page,
    filter_units,
)
from wikimeasurements.utils.nlp_utils import (
    get_amods_and_compounds,
    custom_sentencizer,
    lower_case_lemmas,
    apply_spacy_matcher,
    visualize_annotations,
    find_datetimes,
    extract_values_from_span,
    match_adjecent_noun_as_unit,    
)
from wikimeasurements.utils.quantity_utils import (
    str2num,
    get_number_words_quantifier_and_physical_constants,
    MAPE,
    is_small_int,
    convert_temperature,
    get_basic_number_words,
    get_frequent_units,
)
from wikimeasurements.utils.wikidata_utils import load_wikidata_knowledge
from wikimeasurements.utils.general_utils import (
    init_logger,
    compare_datetimes,    
)
from wikimeasurements.utils.type_aliases import (
    Offset,
    Annotations,    
    Labels,
    UnitConvData,    
)
import coreferee
import spacy
from spacy.tokens import Token
from spacy.tokens.span import Span
from spacy.tokens.doc import Doc
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from bisect import bisect_left
from dateparser import parse as date_parser


# Just use "tok2vec", "tagger", "attribute_ruler", "lemmatizer"
# and "lower_lemmas" for faster lemmatization.
COMPS_NOT_REQUIRED_FOR_EN_LEMMATIZER = [
    "senter",
    "custom_sentencizer",
    "parser",
    "ner",
    "coreferee",
    "merge_entities",
]

POSTPRO_PATTERN = (
    r"(^,?\s+)|"  # Strip leading whitespace and ", " if sentence is wrongly split
)
POSTPRO_PATTERN += r"(\s+$)|"  # Strip trailing whitespace
POSTPRO_PATTERN += r"(\[\[)|(\]\])|"  # Remove all double square brackets
POSTPRO_PATTERN += r"( \( , , \))"  # Remove " ( , , )"
POSTPRO_PATTERN = re.compile(POSTPRO_PATTERN)


def postprocess_sentence(s_text: str, quantity_annotations, debug_mode: bool = False):
    """Post-process a sentence by removing trailing and leading whitespace, 
    removing all double square brackets, and empty parantheses.    
    """

    # Get substrings that should be removed
    pp_matches = [match.span() for match in POSTPRO_PATTERN.finditer(s_text)]

    # Sort and reverse list
    pp_matches = sorted(pp_matches, key=lambda x: x[1], reverse=True)

    # Remove substrings and adapt char offsets of quantity annotations
    pp_s_text = s_text
    pp_quantity_annotations = quantity_annotations.copy()
    for match in pp_matches:

        # Remove substring
        pp_s_text = pp_s_text[: match[0]] + pp_s_text[match[1] :]

        # Adapt char offsets
        char_span = match[0] - match[1]
        for i, (start, end) in enumerate(pp_quantity_annotations):

            if end > match[0]:
                if end > match[1]:
                    # Set new annotation end offset
                    end += char_span

                    # Set new annotation start offset
                    if start > match[1]:
                        start += char_span
                    elif start > match[0]:
                        start = match[0]
                else:
                    # Set new annotation end offset
                    end = match[0]

                    # Set new annotation start offset
                    if start > match[0]:
                        start = match[0]

            if debug_mode and (
                s_text[quantity_annotations[i][0] : quantity_annotations[i][1]]
                != pp_s_text[start:end]
            ):
                raise ValueError("Quantity annotation string changed during post-processing. Investigate!")

            pp_quantity_annotations[i] = (start, end)

    return pp_s_text, pp_quantity_annotations


def get_quantity_annotations(
    nlp: Language,
    doc: Doc,
    quantities,
    other_text: list[str],
    NUM_WORD_BLACKLIST,
    debug_mode: bool = False,
):

    examples = []
    dropped_examples = []

    # Get quantity annotation offsets
    text_lengths = [len(t) for t in other_text]
    quantity_lengths = [len(q) for q in quantities]
    quantity_offsets = []
    start_char = 0
    end_char = 0
    for i, q_len in enumerate(quantity_lengths):
        start_char = end_char + text_lengths[i]
        end_char = start_char + q_len
        quantity_offsets.append((start_char, end_char))

    # Check if sentence includes a quantity annotation and
    # that no numbers which might be part of other quantities
    # occur outside the quantity annotations
    for sent in doc.sents:

        if sent.text == "\n\n" or sent[0:2].text == "##" or len(sent) < 3:
            # Is only linebreaks, seems to be a header
            # or too short, thus skip.
            continue

        s_start = sent.start_char
        s_end = sent.end_char
        s_text = sent.text

        # Check for quantity annotations
        quantity_annotations = []
        for q_start, q_end in quantity_offsets:
            if s_start <= q_start and s_end >= q_end:
                # Add annotation
                q_annotation = (q_start - s_start, q_end - s_start)
                quantity_annotations.append(q_annotation)

        if len(quantity_annotations) > 0:
            # Check for numbers outside the quantity annotations
            (
                is_valid_sentence,
                filter_matches,
                kept_based_on_ner,
                greedy_only_matches,
            ) = filter_sentence(
                nlp,
                sent,
                quantity_annotations,
                NUM_WORD_BLACKLIST,
                debug_mode=debug_mode,
            )

            # Post-processing, that is, remove trailing and
            # leading whitespace and some other patterns.
            s_text, quantity_annotations = postprocess_sentence(
                s_text, quantity_annotations, debug_mode=debug_mode
            )

            # Visualize annotations by enclosing them with colorful fruits.
            if debug_mode:
                ann_tags = [
                    (quantity_annotations, "🍏"),
                    (filter_matches, "🍎"),
                    (kept_based_on_ner, "🍋"),
                    (greedy_only_matches, "🍇"),
                ]
                s_text_ann = visualize_annotations(s_text, ann_tags)
                print(s_text_ann)

            # Create example dict for output
            example = (
                {
                    "sentence": s_text_ann if debug_mode else s_text,
                    "quantities": quantity_annotations,
                },
            )

            # Add sentence and annotation to list of examples to
            # add to the dataset or dropped examples for debugging.
            if is_valid_sentence:
                examples.append(example)
            else:
                dropped_examples.append(example)

    if len(examples) == 0 and len(dropped_examples) == 0:
        logger.warning("Neither got an example nor dropped an example. Probably split sentence within a quantity.")

    return examples, dropped_examples


def match_qualifiers(
    sent: Span,
    doc: Doc,
    fact: dict,
    LABELS: Labels,
    nlp: Language,
    stats: dict[str, defaultdict[int]],
    exclude_spans: Union[tuple[int, int, str], None] = None,
    sent_amods: Union[defaultdict[Token, list[Token]], None] = None,
    debug_mode: bool = False,
) -> Annotations:
    """Annotate qualifiers of quantitative statements in text
        based on facts from a knowledge base. A qualifier
        constrains a quantitative statement, for example, in
        terms of the measurement method, date, precision etc.

    :param sent: a sentence to analyze
    :type sent: spacy.tokens.span.Span
    :param fact: facts from knwoledge base
    :type fact: dict
    :return: annotations
    :rtype: dict
    """

    if fact["qualifiers"] == [""]:
        # No qualifier, no annotations.
        return {}, stats

    # Qualifiers to skip
    skip_those_qualifiers = {"P1352": "ranking"}

    # Token matching
    # e.g., criterion_used can have value estimation (Q791801)
    textual_qualifiers = {
        "P518": "applies_to_part",
        "P642": "in_scope_of",
        "P276": "location",
        "P1013": "criterion_used",
        "P459": "determination_method",
        "P3680": "according_to",
    }

    # Examples of qualfiers not in map
    #   sex_or_gender = 'http://www.wikidata.org/prop/qualifier/P21'
    #   astronomical_filter = 'http://www.wikidata.org/prop/qualifier/P1227'
    #   legislative body = 'http://www.wikidata.org/prop/qualifier/P194'

    # Time matching
    time_qualifiers = {
        "P585": "point_in_time",
        "P580": "start_time",
        "P582": "end_time",
    }

    # Map codes for time precisions to something readable.
    # Note that we do not support precisions greater than years.
    prec_map = {
        None: None,
        "0": "year",  # "billion_years",
        "1": "year",  # "hundred_million_years",
        "3": "year",  # "million_years",
        "4": "year",  # "hundred_thousand_years",
        "5": "year",  # "ten_thousand_years",
        "6": "year",  # "millennium",
        "7": "year",  # "century",
        "8": "year",  # "decade",
        "9": "year",
        "10": "month",
        "11": "day",
        "12": "hour",
        "13": "minute",
        "14": "second",
    }

    geo_coordinate_qualifiers = {"P625": "coordinate_location"}

    qualifier_annotations = defaultdict(list)
    for q_uri, qvalue, qv_lb, qv_ub, qunit, qtp in zip(
        fact["qualifiers"],
        fact["qualifier_values"],
        fact["qualifier_lowerbounds"],
        fact["qualifier_upperbounds"],
        fact["qualifier_units"],
        fact["qualifier_time_precisions"],
    ):
        if q_uri is None:
            continue
        else:
            qualifier = q_uri.removeprefix("http://www.wikidata.org/prop/qualifier/")

        if qualifier in skip_those_qualifiers.keys():
            continue
        elif qualifier in geo_coordinate_qualifiers.keys():

            qualifier_label = geo_coordinate_qualifiers[qualifier]

            def match_geo_coordinates(geo_point: str):
                """geo_point is string like 'Point(6.1937594 50.6636676)'"""
                (lat, lon) = geo_point.split(" ")
                lat = lat.removeprefix("Point(").split(".")
                lon = lon.removesuffix(")").split(".")

                def match_coordinate(
                    full_degrees: str, decimal_degrees: str, text: str
                ):
                    # Different rounding levels are used. Excerpts from Wikipedia articles:
                    #   * "[...] located at (39.3111112, −94.9224637) [...]"
                    #   * "[...] located at (36.7118, -110.2505) [...]"
                    #   * "[...] located at (34.069719, -112.139466) [...]"
                    # Therefore, round to 3 decimal places and allow a maximum of 9 decimal places.
                    boundary = r"(?:^|(?<=[\s(]))"
                    regex = (
                        boundary
                        + r"("
                        + full_degrees
                        + "."
                        + decimal_degrees[:3]
                        + r"\d{0,6})"
                    )
                    matches = re.findall(regex, text)

                    annotation_offsets = [match.span() for match in matches]

                    return annotation_offsets

                lat_offsets = match_coordinate(lat[0], lat[1], sent.text)
                lon_offsets = match_coordinate(lon[0], lon[1], sent.text)

                return lat_offsets, lon_offsets

            lat_offsets, lon_offsets = match_geo_coordinates(qvalue)

            if len(lat_offsets) == 1 and len(lon_offsets) == 1:
                qualifier_annotations[qualifier_label].append(char_offsets)

        elif qualifier in time_qualifiers.keys():

            qualifier_label = time_qualifiers[qualifier]

            # Get dates to compare
            # TODO: Set the default timezone to UTC−00:00 for London local time
            known_date = date_parser(qvalue, languages=["en"])
            try:
                candi_dates = find_datetimes(sent, nlp)
            except RuntimeError:
                # This error occured multiple times:
                # "[E039] Array bounds exceeded while searching for root word.
                #  This likely means the parse tree is in an invalid state..."
                continue

            # Get precision
            prec = prec_map.get(qtp)

            # Compare the date from Wikidata with the dates in the text
            for candi_date in candi_dates:
                dates_match = compare_datetimes(
                    known_date, candi_date.text, known_prec=prec
                )
                if dates_match:
                    char_offsets = (candi_date.start_char, candi_date.end_char)
                    qualifier_annotations[qualifier_label].append(char_offsets)

        else:

            # Get above-specified label or generic label "some_qualifier"
            qualifier_label = textual_qualifiers.get(qualifier, "some_qualifier")

            # Get aliases
            qvalue_aliases = LABELS["qualifiers_values"].get(qvalue)
            qunit_aliases = LABELS["qualifiers_units"].get(qunit)

            if (qvalue_aliases is not None) or (qunit_aliases is not None):

                # Get matcher
                qualifier_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA", validate=debug_mode)

                with nlp.select_pipes(disable=COMPS_NOT_REQUIRED_FOR_EN_LEMMATIZER):
                    if qvalue_aliases is not None:
                        qvalue_docs = list(nlp.pipe(qvalue_aliases))
                        qualifier_matcher.add(qualifier_label, qvalue_docs)

                    if qunit_aliases is not None:
                        qunit_docs = list(nlp.pipe(qunit_aliases))
                        qualifier_matcher.add(qualifier_label, qunit_docs)

                matches, stats, _, _ = apply_spacy_matcher(
                    qualifier_matcher,
                    sent,
                    doc,
                    nlp,
                    stats,
                    [],
                    exclude_spans=exclude_spans,
                    expand_to_amods=True,
                    amods=sent_amods,
                )

                for tag, char_offsets in matches.items():
                    qualifier_annotations[tag] += char_offsets

    return qualifier_annotations, stats


def match_known_value(
    number_offsets: list[Offset],
    value: Decimal,
    value_lb: Union[Decimal, None],
    value_ub: Union[Decimal, None],
    sent: Span,
    stats: dict[str, defaultdict[int]],
    weak_accept_reasons: list[str],
    threshold: float = 0.03,
) -> tuple[list[Offset], list[Union[int, float]], list[float]]:
    """Parse all identified number strings into numeric datatypes
    and compare their value against the fact's numeric value.

    We allow for approximate matches, to allow matches like this:
      Wikidata: <Alabama, life expectancy, 75.5 year, point in time: 2018>
      Wikipedia: In 2018, life expectancy in Alabama was 75.1 years
    """
    value_annotations = []
    value_nums = []
    deviations = []
    for (start, end) in number_offsets:

        match_value_str = sent.text[start:end]
        match_value = str2num(match_value_str)

        if match_value is not None:
            target_value = float(value)
            deviation = MAPE(target_value, match_value)
            if None not in [value_lb, value_ub]:
                # If a lower and upper bound is provided,
                # we accept all values within this range.
                values_match = value_lb <= value <= value_ub
                deviation = 0 if values_match else deviation
            elif deviation <= threshold:
                # Check if deviation between values is smaller than a threshold.
                values_match = True
            elif abs(target_value) < 0.5 / threshold:
                # Consider rounding of decimal places of small numbers can lead to
                # deviations higher than the threshold (e.g., MAPE(1.2, 1.15) > 0.03)
                n_dec_target = value.as_tuple()
                target_decimal_places = -n_dec_target.exponent
                try:
                    n_dec_match = Decimal(match_value).as_tuple()
                    match_decimal_places = -n_dec_match.exponent
                except:
                    match_decimal_places = target_decimal_places

                if target_decimal_places != match_decimal_places:
                    prec = min(target_decimal_places, match_decimal_places)
                    if prec == 0 and ((-1 < target_value < 1) or (-1 < match_value < 1)):
                        # We want to round 1.234 to 1 but not 0.123 to 0
                        prec = 1

                    rounded_deviation = MAPE(round(target_value, prec), round(match_value, prec))
                    if rounded_deviation <= threshold:
                        stats["accept_cause"]["relax_threshold_for_small_rounded_values"] += 1
                        weak_accept_reasons.append("relax_threshold_for_small_rounded_values")
                        values_match = True
                    else:
                        values_match = False
                else:
                    values_match = False
            else:
                values_match = False

            if values_match:
                deviations.append(deviation)
                value_nums.append(match_value)
                value_annotations.append((start, end))

    return value_annotations, value_nums, deviations, stats, weak_accept_reasons


def get_fulltext(page: dict, shorten: bool = False) -> tuple[str, list[str], list[str]]:
    """Build a continous text from alternating lists of quantities and other text."""

    quantities = []
    for q in page["quantities"]:
        if q is None:
            quantities.append("")
        else:
            # Three quantitiy represantations exist:
            # "6,992 km", "4,345 mi" and "6,992 km (4,345 mi)"
            # We choose one by random but weight the third option lower.
            quantities.append(q[choices([0, 1, 2], [0.475, 0.475, 0.05])[0]])

    # Get text and normalize unicode chars (e.g., '\xa0' to ' ')
    other_text = [unicodedata.normalize("NFKD", "".join(t)) for t in page["text"]]

    if shorten:
        # Safely delete sentences with no {{convert}} quantity annotation
        # inside by assuming double newlines commonly used for seperating
        # paragraphs as a safe boundary between two sentences and keeping
        # only the first and last paragraph.
        shortened_other_text = []
        for t in other_text:
            t_split = t.split("\n\n")
            first_and_last_par = t_split[0]  # Add first paragraph
            if len(t_split) > 1:
                # Add last parargraph.
                first_and_last_par += "\n\n" + t_split[-1]

            shortened_other_text.append(first_and_last_par)
        other_text = shortened_other_text

    # Get full text
    full_text = [None] * (len(quantities) + len(other_text))
    full_text[::2] = other_text
    full_text[1::2] = quantities
    full_text = "".join(full_text)

    return full_text, quantities, other_text


def get_context_annotations(
    nlp: Language,
    doc: Doc,
    page: dict,
    LABELS: Labels,
    UNIT_CONVERSION_DATA: UnitConvData,
    frequent_units_matcher: PhraseMatcher,
    WIKI_ORDINAL_LEMMAS: list[int],
    WIKI_ORDINALS_LOWER: list[str],
    debug_mode: bool = False,
):
    """Match quantitative statements from Wikidata with their associated
    Wikipedia article in order to create a dataset of quantities with
    their measurement context annotated in text.

    At least the numeric value, unit, entity and property of the given
    fact must be aligned to the given sentence to form a valid training
    example.
    """

    approx_value_match_threshold = 0.03
    conv_factor_ub = 2_000
    conv_factor_lb = 1 / conv_factor_ub
    wikidata_entity_root = "http://www.wikidata.org/entity/"
    
    page_examples = defaultdict(list)
    stats = {
        "dropped_cause": defaultdict(int),
        "accept_cause": defaultdict(int),
        "counts": defaultdict(int),
    }

    # Get information about the page
    sent_text_list = []
    amods_list = []
    par_starts = []
    all_number_offsets = []
    coref_cache = {}
    sent_list = list(doc.sents)

    for i, sent in enumerate(doc.sents):

        # Create lists of sentences and adjectival modifiers (amods) within
        sent_text_list.append(sent.text)
        amods_list.append(get_amods_and_compounds(sent, only_consecutive=True))

        # Get paragraph starts
        if (sent.text == "\n\n") or (i == 0):
            par_starts.append(i)
            all_number_offsets.append([])
        else:
            # Identify all numbers in current sentence
            all_number_offsets.append(
                extract_values_from_span(
                    sent,
                    doc,
                    consider_ordinals=False,
                    WIKI_ORDINAL_LEMMAS=WIKI_ORDINAL_LEMMAS,
                    WIKI_ORDINALS_LOWER=WIKI_ORDINALS_LOWER,
                )
            )

    for fact in page["wikidata_facts"]:

        # We anchor patterns at the numeric value, since
        # entities are typically all over the place in
        # their respective Wikipedia articles and the
        # properties are sometimes implicit.
        initial_value_str = fact["value"]
        initial_value_lb_str = fact["value_lowerbound"]
        initial_value_ub_str = fact["value_upperbound"]
        initial_unit = "wd:" + fact["unit"].removeprefix(wikidata_entity_root)

        # Get aliases
        entity_aliases = LABELS["entities"].get(fact["entity"])
        property_aliases = LABELS["properties"].get(fact["property"])
        if None in [entity_aliases, property_aliases]:
            stats["dropped_cause"]["missing_alias_for_entity_or_property"] += 1            
            continue

        # Get matcher: We use a second matcher since unlike for value
        # matching we can't speed up the process by matching on ORTH
        # which only requrires running the tokenizer.
        context_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA", validate=debug_mode)

        with nlp.select_pipes(disable=COMPS_NOT_REQUIRED_FOR_EN_LEMMATIZER):
            entities = list(nlp.pipe(entity_aliases))
            props = list(nlp.pipe(property_aliases))
            context_matcher.add("ENTITY", entities)
            context_matcher.add("PROPERTY", props)

        # Unit conversion allows for matches like this:
        #   Wikidata: <Alabama, area, 134000 square kilometre>
        #   Wikipedia: "🌶️Alabama🌶️ is the thirtieth-largest state in the United States
        #               with 🍏52,419🍏 🍓square miles🍓 of 🍊total 🍊area🍊🍊"

        # Get units to consider for conversion.
        # Looping over all known units the given unit can be converted to
        # would yield in a bad performance, because, for example, metre
        # can be converted into 231 other units based on Wikidata.
        # Dropping units for which the conversion factor is out of a
        # threshold bound of 1/1000 and 1000 and for which no label exists
        # reduces the number to 119.

        if initial_unit in ["wd:Q25267", "wd:Q11579", "wd:Q42289"]:
            # Get conversion data for converting between Kelvin,
            # degree Celsius and degree Fahrenheit
            considered_units = convert_temperature(initial_value_str, initial_unit)
            if not None in [initial_value_lb_str, initial_value_ub_str]:
                values_ub = convert_temperature(initial_value_lb_str, initial_unit)
                values_lb = convert_temperature(initial_value_ub_str, initial_unit)
        else:
            considered_units = []
            for unit in UNIT_CONVERSION_DATA[initial_unit]:
                # Get conversion factor to convert unit from "initial_unit" to "unit"
                conv_factor = Decimal(unit["conv_factor_numerator"]) / Decimal(unit["conv_factor_denominator"])

                # Check if unit should be considered or neglected to speed up the process
                unit_has_label = LABELS["units"].get(wikidata_entity_root + unit["to_unit"]) != None
                if unit_has_label:
                    # Get criteria to determine if unit conversion is common
                    is_power_of_ten = math.log10(conv_factor) % 1 == 0
                    within_thresholds = conv_factor_lb <= conv_factor <= conv_factor_ub
                    if is_power_of_ten or within_thresholds:
                        # Unit should be considered!
                        considered_units.append((unit, conv_factor))

        # Debug-tip: Get all considered unit labels
        # [LABELS["units"].get(wd_entity_root + u["to_unit"]) for u in considered_units]

        for unit_data, conv_factor in considered_units:

            # Prepare unit matcher
            if unit_data["to_unit"] == "Q199":
                # Q199 is the the Wikidata item for the natural number 1
                # It is used as a unit for counts. We do not need to match it.
                unit_matcher = None
                unit_aliases = []
            else:
                unit = wikidata_entity_root + unit_data["to_unit"]
                unit_matcher = PhraseMatcher(
                    nlp.vocab, attr="LEMMA", validate=debug_mode
                )
                unit_aliases = LABELS["units"].get(unit)

                if unit_aliases is None:
                    unit_matcher = None
                    unit_aliases = []
                else:
                    with nlp.select_pipes(disable=COMPS_NOT_REQUIRED_FOR_EN_LEMMATIZER):
                        units = list(nlp.pipe(unit_aliases))
                        unit_matcher.add("UNIT", units)

            if unit_data["to_unit"] in ["Q25267", "Q11579", "Q42289"]:
                # For temperatures the calculation of converted values differs
                value = unit_data["converted_temp_value"]
            else:
                # Convert value to current unit using a known conversion factor
                conv_factor = Decimal(unit_data["conv_factor_numerator"]) / Decimal(unit_data["conv_factor_denominator"])
                value = Decimal(initial_value_str) * conv_factor

            # If avaivable, get lower and upper bound
            if None in [initial_value_lb_str, initial_value_ub_str]:
                value_ub = None
                value_lb = None
            elif unit_data["to_unit"] in ["Q25267", "Q11579", "Q42289"]:
                value_ub = None
                value_lb = None
                for t_ub, t_lb in zip(values_ub.values(), values_lb.values()):
                    if t_ub["to_unit"] == unit_data["to_unit"] and t_lb["to_unit"] == unit_data["to_unit"]:
                        value_ub = t_ub["converted_temp_value"]
                        value_lb = t_lb["converted_temp_value"]
                        break
            else:
                value_ub = Decimal(initial_value_ub_str) * conv_factor
                value_lb = Decimal(initial_value_lb_str) * conv_factor

            for sent_idx, sent in enumerate(doc.sents):

                if sent_idx in par_starts or len(sent) < 3:
                    # Do not analyze "\n\n" or very very short sentences.
                    continue

                weak_accept_reasons = []

                # Get all numbers in current sentence
                number_offsets = all_number_offsets[sent_idx]

                # Parse all identified number strings into numeric datatypes
                # and compare their value against the fact's numeric value.
                (
                    value_annotations,
                    value_nums,
                    deviations,
                    stats,
                    weak_accept_reasons,
                ) = match_known_value(
                    number_offsets,
                    value,
                    value_lb,
                    value_ub,
                    sent,
                    stats,
                    weak_accept_reasons,
                    threshold=approx_value_match_threshold,
                )

                (
                    value_annotations,
                    deviations,
                    stats,
                    discard,
                ) = filter_value_annotations(sent, value_annotations, deviations, unit_matcher, stats)

                if discard:
                    continue

                # all_annotations_flattened = value_annotations.copy()
                all_annotations_flattened = [(a, b, "VALUE") for (a, b) in value_annotations]

                small_int_count = False
                if unit_matcher is None:
                    is_count = True
                    # Check if either the given or matched quantity is a small
                    # integer count, since those require stricter filtering.
                    small_int_count = is_small_int(value, threshold=10) or is_small_int(value_nums[0], threshold=10)

                    # For counts take nouns succeeding the value as unit
                    unit_annotations, stats, discard = match_adjecent_noun_as_unit(
                        value_annotations, sent, doc, frequent_units_matcher, stats
                    )

                    if discard:
                        continue

                else:
                    # Identify unit
                    is_count = False
                    (
                        unit_annotations,
                        stats,
                        weak_accept_reasons,
                        _,
                    ) = apply_spacy_matcher(
                        unit_matcher,
                        sent,
                        doc,
                        nlp,
                        stats,
                        weak_accept_reasons,
                        exclude_spans=all_annotations_flattened,
                        expand_to_amods=False,
                        expand_to_noun_chunk=False,
                        expand_to_ner_annotation=False,
                    )

                unit_annotations, stats, discard = filter_unit_annotations(
                    sent,
                    doc,
                    unit_annotations,
                    is_count,
                    value_annotations,
                    number_offsets,
                    frequent_units_matcher,
                    stats,
                )
                if discard:
                    continue

                all_annotations_flattened += [
                    (a, b, "UNIT") for (a, b) in unit_annotations["UNIT"]
                ]

                sent_amods = amods_list[sent_idx]

                par_end_idx = bisect_left(par_starts, sent_idx)
                par_start_idx = max(0, par_end_idx - 1)
                coref_context = sent_list[
                    min(par_starts[par_start_idx] + 1, sent_idx) : sent_idx + 1
                ]

                # Get measured entity and measured property annotation
                (
                    context_annotations,
                    stats,
                    weak_accept_reasons,
                    coref_cache,
                ) = apply_spacy_matcher(
                    context_matcher,
                    sent,
                    doc,
                    nlp,
                    stats,
                    weak_accept_reasons,
                    exclude_spans=all_annotations_flattened,
                    expand_to_amods=True,
                    amods=sent_amods,
                    frequent_units_matcher=frequent_units_matcher,
                    expand_to_ner_annotation=True,
                    expand_to_propn=True,
                    check_corefs=True,
                    coref_context=coref_context,
                    coref_cache=coref_cache,
                    is_count=is_count,
                    deviations=deviations,
                )

                (
                    context_annotations,
                    stats,
                    discard,
                    weak_accept_reasons,
                ) = filter_context_annotations(
                    context_annotations,
                    value_annotations,
                    unit_annotations,
                    sent,
                    doc,
                    nlp,
                    fact["property"],
                    LABELS,
                    stats,
                    value,
                    is_count,
                    deviations,
                    weak_accept_reasons,
                    debug_mode=debug_mode,
                )
                if discard:
                    continue

                # Get qualifiers
                all_annotations_flattened += [
                    (a, b, "PROPERTY") for (a, b) in context_annotations["PROPERTY"]
                ]
                all_annotations_flattened += [
                    (a, b, "ENTITY") for (a, b) in context_annotations["ENTITY"]
                ]

                qualifier_annotations, stats = match_qualifiers(
                    sent,
                    doc,
                    fact,
                    LABELS,
                    nlp,
                    stats,
                    exclude_spans=all_annotations_flattened,
                    sent_amods=sent_amods,
                    debug_mode=debug_mode,
                )

                nbr_qualifier_ann = sum(
                    [len(q) for q in qualifier_annotations.values()]
                )
                if small_int_count and nbr_qualifier_ann == 0:
                    # Small integer counts (that is, without units) are
                    # frequent and thus require higher confidence.
                    stats["dropped_cause"]["small_int_count_without_qualifiers"] += 1                    
                    continue

                # Add stats about unit conversion factor.
                if debug_mode:
                    if conv_factor == None:
                        stats["accept_cause"]["conversion_of_temperatures"] += 1
                    else:
                        stats["accept_cause"]["conv_factor_is_" + str(conv_factor)] += 1

                annotations = {
                    # Quantitative statement
                    "entity": context_annotations["ENTITY"],
                    "property": context_annotations["PROPERTY"],
                    "value": value_annotations,
                    "unit": unit_annotations["UNIT"],
                    # Temporal scope
                    "point_in_time": qualifier_annotations.get("point_in_time", []),
                    "start_time": qualifier_annotations.get("start_time", []),
                    "end_time": qualifier_annotations.get("end_time", []),
                    # Spatial scope
                    "location": qualifier_annotations.get("location", []),
                    "coordinate_location": qualifier_annotations.get(
                        "coordinate_location", []
                    ),
                    # Item scope
                    "applies_to_part": qualifier_annotations.get("applies_to_part", []),
                    "in_scope_of": qualifier_annotations.get("in_scope_of", []),
                    # Source and method
                    "criterion_used": qualifier_annotations.get("criterion_used", []),
                    "determination_method": qualifier_annotations.get(
                        "determination_method", []
                    ),
                    "according_to": qualifier_annotations.get("according_to", []),
                    # Other qualifiers
                    "some_qualifier": qualifier_annotations.get("some_qualifier", []),
                }

                if debug_mode:
                    # Visualize annotations by enclosing them with colorful fruits
                    ann_tags = [
                        (annotations["entity"], "🌶️"),
                        (annotations["property"], "🍊"),
                        (annotations["value"], "🍏"),
                        (annotations["unit"], "🍓"),
                        (annotations["point_in_time"], "📆"),
                        (annotations["start_time"], "⏱️"),
                        (annotations["end_time"], "⏰️"),
                        (annotations["location"], "📍"),
                        (annotations["coordinate_location"], "📍"),
                        (annotations["applies_to_part"], "🦵"),
                        (annotations["in_scope_of"], "🔎"),
                        (annotations["criterion_used"], "📏"),
                        (annotations["determination_method"], "🔭"),
                        (annotations["according_to"], "🙋"),
                        (annotations["some_qualifier"], "🛁"),
                    ]
                    s_text_ann = visualize_annotations(sent.text, ann_tags)

                # Get all sentences of the paragraph to the left and right of the sentence
                sents_before = sent_text_list[
                    min(par_starts[par_start_idx] + 1, sent_idx) : sent_idx
                ]

                if par_end_idx == len(par_starts):
                    # Sentence is in last paragraph
                    sents_after = sent_text_list[sent_idx + 1 :]
                else:
                    # Sentence is in a paragraph which is followed by another paragraph
                    sents_after = sent_text_list[sent_idx + 1 : par_starts[par_end_idx]]

                # Add document and annotations to training examples
                actual_annotations = {
                    k: v for (k, v) in annotations.items() if len(v) > 0
                }
                example = {
                    "sentence": s_text_ann if debug_mode else sent.text,
                    "annotations": actual_annotations,
                    "wikidata_fact": fact,
                    "context": {"before": sents_before, "after": sents_after},
                    "weak_accept_reasons": weak_accept_reasons,
                }

                page_examples[sent_idx].append(example)

    stats["counts"]["number_of_facts"] = len(page["wikidata_facts"])
    stats["counts"]["analyzed_pages"] = 1
    if len(page_examples) > 0:
        filtered_page_examples, stats = filter_examples_from_page(page_examples, stats)

        if debug_mode:
            print("\n\nRaw results:")
            [[print("* " + e["sentence"]) for e in s] for s in page_examples.values()]
            print("\nFiltered results:")
            [print("* " + e["sentence"]) for e in filtered_page_examples]

        stats["counts"]["number_of_examples"] = len(filtered_page_examples)
        return filtered_page_examples, stats

    else:
        stats["counts"]["number_of_examples"] = 0
        return [], stats


def process_single_page(
    page: dict,
    process_quantities: bool,
    process_context: bool,
    nlp: Language,
    LABELS: Labels,
    UNIT_CONVERSION_DATA: UnitConvData,
    frequent_units_matcher: PhraseMatcher,
    WIKI_ORDINAL_LEMMAS: list[int],
    WIKI_ORDINALS_LOWER: list[str],
    NUM_WORD_BLACKLIST: list[str],
    debug_mode: bool = False,
) -> tuple[dict, list]:

    # Debug tip:
    # if page["title"] != "Columbus, Indiana":
    #     return [], []
    # else:
    #     print("Investigate!")

    # Init output
    examples = {"id": page["id"], "title": page["title"]}

    # Get text of the Wikipedia articles and the quantities within.
    full_text, quantities, other_text = get_fulltext(
        page, shorten=not process_context
    )

    # Determine which components to disable temporally to be faster!
    if not process_context:
        # We just do sentence boundary detection based on the
        # dependency parser and a custom rule. For this we
        # keep "tok2vec", "custom_sentencizer" and "parser".
        disable_comps = [
            "senter",
            "tagger",
            "attribute_ruler",
            "lemmatizer",
            "ner",
            "coreferee",
            "merge_entities",
            "lower_lemmas",
        ]
    else:
        disable_comps = ["merge_entities", "coreferee"]

    # Apply spaCy NLP pipeline.
    with nlp.select_pipes(disable=disable_comps):
        doc = nlp(full_text)

    # Create training examples for a dataset of sentences where a single
    # quantity is annotated alongside its measurement context, that is,
    # the measured entity, property and qualifiers.
    # (e.g., "The 🦵highest point🦵 in 🌶️Aachen🌶️ has an 🍊elevation🍊 of around 🍏400🍏 🍓m🍓 above mean sea level.")
    if process_context:
        context_examples, context_stats = get_context_annotations(
            nlp,
            doc,
            page,
            LABELS,
            UNIT_CONVERSION_DATA,
            frequent_units_matcher,
            WIKI_ORDINAL_LEMMAS,
            WIKI_ORDINALS_LOWER,
            debug_mode=debug_mode,
        )
    else:
        context_stats = {
            "dropped_cause": {},
            "accept_cause": {},
            "counts": {},
        }
        context_examples = []

    # Create training examples for a dataset of sentences where all quantities are
    # annotated based on the usage of the {{convert}} template in Wikipedia articles.
    # (e.g., "Most fountains are in either the Valley, about 🍏50 miles🍏 northeast, 
    # or on the Peninsula, approx. 🍏90 miles🍏 southwest.")
    if process_quantities:
        quantity_examples, dropped_quantity_examples = get_quantity_annotations(
            nlp, doc, quantities, other_text, NUM_WORD_BLACKLIST, debug_mode
        )
        quantity_stats = {"counts": {"number_of_examples": len(quantity_examples)}}
    else:
        quantity_stats = {"counts": {"number_of_examples": 0}}
        quantity_examples = []
        dropped_quantity_examples = []

    # Add generated examples
    examples.update({"context_dataset": context_examples})
    examples.update({"quantity_dataset": quantity_examples})

    # Merge stats
    stats = {
        "context_dataset": context_stats,
        "quantity_dataset": quantity_stats,
    }

    return examples, stats, dropped_quantity_examples


def get_stats_for_progress_bars(parsed_dump_path: str, nbr_facts: int):
    """Get number of pages for which an output, that is,
    an entry in the dataset, is expected.
    """
    if args.output_mode == "debug":
        # Just make up some numbers to be faster.
        total_page_count = 123456789
        num_pages_using_cvt = 1234567
        num_pages_with_output_expected = 123456

    else:
        with open(parsed_dump_path) as f:
            # Get total number of pages and number of pages using
            # the convert template in the Wikipedia dump.
            total_page_count = 0
            num_pages_using_cvt = 0
            for l in f:
                total_page_count += 1
                if len(json.loads(l).get("quantities")) != 0:
                    num_pages_using_cvt += 1

        # Expected number of pages where we are able to match at least one fact
        approx_example_per_fact_ratio = 1.2 / 100
        num_pages_with_matchable_facts = round(
            nbr_facts * approx_example_per_fact_ratio
        )

        num_pages_with_output_expected = max(
            int(build_quantity_set) * num_pages_using_cvt,
            int(build_context_set) * num_pages_with_matchable_facts,
        )

        logger.info(f"📚 There are {num_pages_using_cvt} pages in {args.input_path} which use the convert template.\n")

    return total_page_count, num_pages_using_cvt, num_pages_with_output_expected


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--omit_quantity_dataset_creation",
        action="store_true",
        help="""Whether the quantity dataset should be created or not.""",
    )
    parser.add_argument(
        "--omit_context_dataset_creation",
        action="store_true",
        help="""Whether the context dataset should be created or not.""",
    )
    parser.add_argument(
        "--input_path",
        default="./workflow/intermediates/parsed_wikipedia_dump.json",
        help="""Path to the Wikipedia dump parsed with parse_wikipedia_dump.py""",
    )
    parser.add_argument(
        "--outfile",
        default="./workflow/output/quantity_dataset.json",
        help="""Path of output file (e.g., './quantity_dataset.json')""",
    )
    parser.add_argument(
        "--wikidata_facts_path",
        default="./workflow/intermediates/wikidata_quantitative_statements.json",
        help="""Path to quantitative statements queried from Wikidata with query_wikidata.py""",
    )
    parser.add_argument(
        "--unit_freqs_path",
        default="./workflow/intermediates/unit_freqs.json",
        help="""Path to unit frequencies file created with get_unit_frequencies.py""",
    )
    parser.add_argument(
        "--log_file",
        default="./logs/create_datasets.log",
        help="""Path of file the workers will log to.""",
    )
    parser.add_argument(
        "--page_offset",
        default=-1,
        help="""Skip all pages until page offset.""",
    )
    
    parser.add_argument(
        "--output_mode",
        default="debug",
        help="""Choose between "debug" and "production". """,
    )

    args = parser.parse_args()
    logger = init_logger(args.log_file, logging.DEBUG)

    build_quantity_set = not args.omit_quantity_dataset_creation
    build_context_set = not args.omit_context_dataset_creation
    
    if not build_quantity_set and not build_context_set:
        logger.info("Nothing to do. Both quantity and context dataset creation was omitted.")
        quit()
    
    # Create gazeteer matcher for alphabetic numbers
    NUM_WORD_BLACKLIST, _ = get_number_words_quantifier_and_physical_constants()

    if not build_context_set:
        nbr_facts = 0
        LABELS = {}
        UNIT_CONVERSION_DATA = ({},)
        FREQUENT_UNITS = []
    else:
        # Get distant supervision knowledge for context dataset
        (
            facts,
            nbr_facts,
            pages_with_facts,
            LABELS,
            UNIT_CONVERSION_DATA,
        ) = load_wikidata_knowledge(args.wikidata_facts_path)

        # Get frequent units
        with open(args.unit_freqs_path) as f:
            UNIT_FREQS = json.load(f)

        FREQUENT_UNITS = get_frequent_units(LABELS, UNIT_FREQS["quantities"])

        # Get unit conversion data
        UNIT_CONVERSION_DATA = filter_units(
            UNIT_CONVERSION_DATA,
            UNIT_FREQS["fulltext"],
            remove_units_without_labels=True,
            remove_infrequent_units=False,
        )

    # TODO: Wait for GPU cluster and then use transformer model
    # spacy.require_gpu()
    # nlp = spacy.load("en_core_web_trf")
    nlp = spacy.load("en_core_web_md")

    # Add coreference resolution component
    # TODO: Switch to spaCy version when merged and documented
    # https://github.com/explosion/spacy-experimental/pull/17
    nlp.add_pipe("coreferee")

    # Merged adjecent tokens of same entity type
    nlp.add_pipe("merge_entities")

    # Spacy provides different options for sentence boundary detection (SBD).
    # We use the dependency parser, since it is the most accurate and we need the
    # dependency tree anyway. In addition, we add special rules using a custom function.
    # (If dependency parse is not required and the pipeline is not transformer-based,
    # senter could be used to speed up SBD, however, it seems to be less accurate. 
    # On the other hand, if speed is not a concern, pySBD (https://github.com/nipunsadvilkar/pySBD) 
    # could be used for more accuracy).
    
    # Force sentence start at double linebreak.
    nlp.add_pipe("custom_sentencizer", before="parser")

    # Force lemmas to be lowercase.
    nlp.add_pipe("lower_lemmas")

    # Get total number of lines
    (
        total_page_count,
        num_pages_using_cvt,
        num_pages_with_output_expected,
    ) = get_stats_for_progress_bars(args.input_path, nbr_facts)

    logger.warning(
        "[i] If you get a PytzUsageWarning for the dateparser package, install dateparser from git, " \
        "implement these changes yourself (https://github.com/scrapinghub/dateparser/pull/1062), " \
        "or wait until the fix is included in a new release.\n"
    )

    debug_mode = True if args.output_mode == "debug" else False

    # Prepare frequent unit matcher to filter out examples, where
    # the value is followed by a unit which is not annotated.
    frequent_units_matcher = PhraseMatcher(nlp.vocab, attr="LOWER", validate=debug_mode)
    frequent_units_matcher.add("FREQUENT_UNIT", list(nlp.tokenizer.pipe(FREQUENT_UNITS)))

    # Get num words
    with nlp.select_pipes(disable=COMPS_NOT_REQUIRED_FOR_EN_LEMMATIZER):
        WIKI_ORDINALS_LOWER = get_basic_number_words(numeral_type="ordinals_minus_denominator_intersection")
        WIKI_ORDINAL_LEMMAS = get_basic_number_words(numeral_type="ordinals")
        WIKI_ORDINAL_LEMMAS = list(nlp.pipe(WIKI_ORDINAL_LEMMAS))
        # Assumes that each num word is composed of only a single token
        WIKI_ORDINAL_LEMMAS = list(set([tokens[0].lemma for tokens in WIKI_ORDINAL_LEMMAS]))

    global_stats = {
        "context_dataset": {
            "dropped_cause": defaultdict(int),
            "accept_cause": defaultdict(int),
            "counts": defaultdict(int),
        },
        "quantity_dataset": {
            "dropped_cause": defaultdict(int),
            "accept_cause": defaultdict(int),
            "counts": defaultdict(int),
        },
    }

    output_dropped_examples = False
    output_file = open(args.outfile, "w", encoding="utf-8")

    # Create quantity dataset from preprocessed XML dump
    # Most quantities are lengths, weights, etc., but also exotic quantities
    # such as "6 standard gravities" are suppoted by the convert module.
    with open(args.input_path) as f:

        pbar = tqdm(total=total_page_count, desc="Processing Wikipedia pages", unit="pages")
        # args.page_offset = 50_000
        
        for line in f:
            if pbar.n >= args.page_offset:

                # Each line corresponds to a single Wikipedia page
                wiki_page = json.loads(line)

                # Determine whether to process the page for quantities and/or context or not.
                process_quantities = build_quantity_set and len(wiki_page["quantities"]) > 0
                process_context = build_context_set and wiki_page["id"] in pages_with_facts                
                
                if process_context or process_quantities:
                        
                        # Add known Wikidata facts                        
                        wiki_page["wikidata_facts"] = facts.get(str(wiki_page["id"])) if process_context else None
                    
                        examples, stats, dropped_quantity_examples = process_single_page(
                            wiki_page,
                            process_quantities,
                            process_context,
                            nlp,
                            LABELS,
                            UNIT_CONVERSION_DATA,
                            frequent_units_matcher,
                            WIKI_ORDINAL_LEMMAS,
                            WIKI_ORDINALS_LOWER,
                            NUM_WORD_BLACKLIST,
                            debug_mode=debug_mode,
                        )

                        if len(examples) > 0:
                            logger.info(f"Created {len(examples)} dataset records.")

                        # Add stats to global stats.
                        for dataset_key, dataset_stats in stats.items():
                            for cat_key, cat_value in dataset_stats.items():
                                for k, v in cat_value.items():
                                    global_stats[dataset_key][cat_key][k] += v                      

                        if examples.get("context_dataset") or examples.get("quantity_dataset"):
                            # Dump dict to JSON string and send to writing process
                            examples_json_str = json.dumps(examples, ensure_ascii=False)                                    
                            output_file.write(examples_json_str + "\n")

                        if output_dropped_examples and debug_mode:
                            for row in dropped_quantity_examples:
                                dropped_example_json_str = json.dumps(row, ensure_ascii=False)
                                with open("dropped_quantity_examples.json", "a") as f:
                                    f.write(dropped_example_json_str + "\n")

            # Update progress bar.
            pbar.update(1)
            
            # Once in a while print stats.
            if pbar.n % 10000 == 0:
                logger.info(f"🧮\tCurrent stats:\n" + json.dumps(global_stats, ensure_ascii=False, indent=4))      

        pbar.close()

    logger.info("Finished all tasks! ✔️")
