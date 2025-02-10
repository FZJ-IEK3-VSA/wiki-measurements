import re
import itertools
from decimal import *
from typing import Union
from collections import defaultdict
from wikimeasurements.utils.nlp_utils import (
    apply_spacy_matcher,
    get_span_distance,
    check_for_aliases_in_parentheses,
    shortest_path,
)
from wikimeasurements.utils.type_aliases import (
    Offset,
    Annotations,    
    Labels,
    UnitConvData,
    UnitFreqs,
)
from spacy.tokens.span import Span
from spacy.tokens.doc import Doc
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from bisect import bisect_right


# All components but "ner".
COMPS_NOT_REQUIRED_FOR_NER = [
    "senter",
    "tok2vec",
    "tagger",
    "custom_sentencizer",
    "parser",
    "attribute_ruler",
    "lemmatizer",
    "coreferee",
    "merge_entities",
    "lower_lemmas",
]

# Create regex matcher for numeric chars
NUM_CHAR_PATTERN = re.compile(r"\d+")

# NER tags used to not filter out sentences even though
# they contain numbers outside the quantity annotations.
NER_WHITELIST = [
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "NORP",
    "ORG",
    "PERSON",
    "PRODUCT",
    "TIME",
    "WORK_OF_ART",
]

def filter_sentence(
    nlp: Language,
    sent: Span,
    quantity_annotations,
    NUM_WORD_BLACKLIST,
    debug_mode: bool = True,
):

    is_valid_sentence = True

    if "File:" in sent.text and "|" in sent.text:
        return False, [], [], []

    # Remove quantity annotations from sentence
    cut_sentence = sent.text
    for ann_start, ann_end in quantity_annotations[::-1]:
        cut_sentence = (
            cut_sentence[:ann_start]
            + " " * (ann_end - ann_start)
            + cut_sentence[ann_end:]
        )

    # Drop examples like "🍏380 t🍏." or "about 🍏29 oz🍏 mass of food per meal each day"
    # or "🍏2.1 million barrels🍏 of crude oil per day".
    # TODO: Expand in simple cases of "🍏2.3 million barrels🍏 per day"
    # TODO: Should the whole interval in "range from 🍏73.4 °F🍏 to 🍏84.2 °F🍏" be labeled as quantity?
    if (
        len(cut_sentence) < 10
        or " per " in cut_sentence
        or cut_sentence.startswith("%")
    ):
        return False, [], [], []

    # Check for numeric chars
    matches = []
    greedy_matches = []
    for match in NUM_CHAR_PATTERN.finditer(cut_sentence):
        matches.append(match.span())

    # Check for alphabetic numbers
    for num in NUM_WORD_BLACKLIST:
        # Pattern is anchored at word boundary in order to not
        # match pa🍎ten🍎ted, h🍎eight🍎, etc.
        for match in re.finditer(r"\b" + num, cut_sentence, re.IGNORECASE):
            matches.append(match.span())

        if debug_mode:
            # Do not anchor pattern at word boundary
            for match in re.finditer(num, cut_sentence, re.IGNORECASE):
                greedy_matches.append(match.span())

    greedy_only_matches = []
    if debug_mode:
        # Check if the greedy method yields more matches
        for match in greedy_matches:
            if match not in matches:
                greedy_only_matches.append(match)

    # Discard matches which are dates, products, etc.
    filtered_matches = []
    kept_based_on_ner = []

    # Get NER predictions for sentence, if not already done.
    sent_as_doc = sent.as_doc()
    if not sent_as_doc.is_nered:
        # Only do NER.
        with nlp.select_pipes(disable=COMPS_NOT_REQUIRED_FOR_NER):
            sent_as_doc = nlp(sent_as_doc)

    for match in matches:
        ents = sent_as_doc.char_span(match[0], match[1], alignment_mode="expand").ents
        if not any(ent.label_ in NER_WHITELIST for ent in ents):
            is_valid_sentence = False
            if debug_mode:
                filtered_matches.append(match)
            else:
                # One example is enough to filter out sentence
                break
        elif debug_mode:
            # Prevented from being filered out based on NER tag
            kept_based_on_ner.append(match)

    return is_valid_sentence, filtered_matches, kept_based_on_ner, greedy_only_matches


def filter_units(
    unit_conv_data: UnitConvData,
    unit_freqs: UnitFreqs,
    remove_units_without_labels: bool = True,
    remove_infrequent_units: bool = False,
) -> UnitConvData:
    """Remove units from conversion data, which occur very infrequently
    or never or do not have a label. Note that rare units are still
    annotated, if its version in the Wikidata fact matches directly,
    that is, without conversion.
    """

    # Collect units to exclude.
    units_to_exclude = []
    if remove_infrequent_units:
        infrequent_units = [
            q
            for q, count in unit_freqs["frequencies_of_units_with_label"]
            if count == 0
        ]
        units_to_exclude += infrequent_units

    if remove_units_without_labels:
        units_to_exclude += unit_freqs["units_without_english_labels"]

    nbr_units_before = len(list(itertools.chain(*[[1 for _ in c] for c in unit_conv_data.values()])))

    # Remove units from unit conversion data.
    for unit_in, conv_data_list in unit_conv_data.items():
        for i, conv_dict in enumerate(conv_data_list):
            if conv_dict["to_unit"] in units_to_exclude:
                del unit_conv_data[unit_in][i]

    nbr_units_after = len(list(itertools.chain(*[[1 for _ in c] for c in unit_conv_data.values()])))

    print(f"Exluded {nbr_units_before-nbr_units_after} units which either occur very infrequently or do not have a label.")

    return unit_conv_data


def filter_shares(sent: Span, char_offset: Offset) -> bool:
    """Filter out sentences with "of which" before any
    of the annotations:

    For example:
     * 🌶️<some_company>🌶️ yielded a 🍊total income🍊 of
       £123.45 billion of which 🍓£🍓🍏12.34 billion🍏 
       came from [...].
     * The business made a 🍊profit🍊 of 🍓£🍓 🍏123 billion🍏, 
       of which 🌶️<some_company>🌶️'s share was £12 billion.
    """
    return sent.text[: char_offset[0]].endswith(" of which ") or sent.text[ : char_offset[0]].endswith(" of which, ")         


def filter_value_annotations(
    sent: Span,
    value_annotations: list[Offset],
    deviations: list[float],
    unit_matcher: PhraseMatcher,
    stats: dict[str, defaultdict[int]],
) -> tuple[list[Offset], bool]:
    """At least one value must match. When multiple values match in the same sentence,
    only one and only one must have zero deviation and must have a unit.
    Additionally, the value is not allowed to be preceeded by a string like "of which".
    """
    discard = False
    if len(value_annotations) == 0:
        # Could not align the numeric value, thus proceed with next sentence
        # stats["dropped_cause"]["no_value"] += 1
        discard = True
    elif len(value_annotations) > 1:
        # Got multiple matching values in the same sentence
        if deviations.count(0) == 1 and unit_matcher is not None:
            # However, there is only one value matching exactly the value
            # from the knowledge base. Additionally, the quantity is not
            # a count, hence the unit matcher will reduce ambiguity further.
            single_exact_value_match_idx = deviations.index(0)
            deviations = [0]
            value_annotations = [value_annotations[single_exact_value_match_idx]]
        else:
            # There is not enough evidence to decice on which value is the correct one.
            stats["dropped_cause"]["multiple_values"] += 1
            discard = True

    if not discard:
        discard = filter_shares(sent, value_annotations[0])
        if discard:
            stats["dropped_cause"]["of_which_before_value_annotation"] += 1

    return value_annotations, deviations, stats, discard


def check_for_another_adjecent_unit(
    sent: Span,
    doc: Doc,
    value_annotations: list[Offset],
    frequent_units_matcher: PhraseMatcher,
) -> bool:
    """Check if there is a unit adjecent to the given value annotation."""
    
    sent_start_char = doc[sent.start].idx
    value_end_char = sent_start_char + value_annotations[0][1]
    sent_end_char = doc[min(sent.end, len(doc) - 1)].idx
    sent_after_value = doc.char_span(value_end_char, sent_end_char, alignment_mode="expand")
    matches = frequent_units_matcher(sent_after_value)
    adjecent_unit = len(matches) > 0 and matches[0][1] == sent_after_value.start

    return adjecent_unit


def filter_unit_annotations(
    sent: Span,
    doc: Doc,
    unit_annotations: Annotations,
    is_count: bool,
    value_annotations: list[Offset],
    number_offsets: list[Offset],
    frequent_units_matcher: PhraseMatcher,
    stats: dict[str, defaultdict[int]],
) -> tuple[Annotations, bool]:
    """There must be at least a single unit or the value must be a count with no
    unit adjecent. If a single unit matches which is not adjecent to the
    target value, it is not allowed to be adjecent to another value nor is
    the target value allowed to be adjecent to another unit. If multiple
    units match, one and only one unit must be adjecent to the value.
    Additionally, the unit is not allowed to be preceeded by a string like
    "of which".
    """

    discard = False

    if len(unit_annotations) == 0:
        # Could not align the unit, thus proceed with next sentence.
        # For counts no unit is required, however, if there is a unit
        # adjecent to the value the example is discarded.
        if is_count and not check_for_another_adjecent_unit(sent, doc, value_annotations, frequent_units_matcher):
            pass
        else:
            # stats["dropped_cause"]["no_unit"] += 1
            discard = True

    elif len(unit_annotations["UNIT"]) == 1:
        if abs(get_span_distance(unit_annotations["UNIT"][0], value_annotations[0])) > 2:
            # Discard unit, if it is adjecent to another value but not the one of interest
            for num in number_offsets:
                if abs(get_span_distance(unit_annotations["UNIT"][0], num)) < 2:
                    stats["dropped_cause"]["unit_adjecent_to_other_value"] += 1
                    discard = True
                    break

            # Discard unit, if it not adjecent to the value but another one is
            if check_for_another_adjecent_unit(sent, doc, value_annotations, frequent_units_matcher) == True:
                stats["dropped_cause"]["value_adjecent_to_other_unit"] += 1
                discard = True

    elif len(unit_annotations["UNIT"]) > 1:
        # Check if one and only one unit annotation is adjecent to the value
        span_distances = []
        for a in unit_annotations["UNIT"]:
            span_distances.append(abs(get_span_distance(a, value_annotations[0])))

        min_dist = min(span_distances)
        if min_dist < 2 and span_distances.count(min_dist) == 1:
            # There is only a single unit adjecent to the value,
            # hence we consider it as the correct unit.
            min_dist_idx = span_distances.index(min_dist)
            unit_annotations["UNIT"] = [unit_annotations["UNIT"][min_dist_idx]]
        else:
            stats["dropped_cause"]["multiple_units_adjecent_to_value"] += 1
            discard = True

    if not discard and len(unit_annotations["UNIT"]) > 0:
        discard = filter_shares(sent, unit_annotations["UNIT"][0])
        if discard:
            stats["dropped_cause"]["of_which_before_unit_annotation"] += 1

    return unit_annotations, stats, discard


def reduce_ambiguity(
    sent: Span,
    annotations: list[Offset],
    stats: dict[str, defaultdict[int]],
    weak_accept_reasons: list[str],
    value: Offset,
    unit: Offset,
    entity: Offset = [],
    property: Offset = [],
):
    """If multiple spans match are candidates for the measured entity or property,
    exclude all but one by applying different heuristics.
    """
    # TODO: Check for patterns like "number property, number property" and "property number, property number".

    # === Check for mention followed by aliases in parentheses ===
    parentheses_exception_applies = check_for_aliases_in_parentheses(sent, annotations)

    if parentheses_exception_applies:
        # Merge annotations
        min_start = min(start for start, end in annotations)
        max_end = max(end for start, end in annotations)
        stats["accept_cause"]["choose_single_annotation_based_on_parentheses"] += 1
        return [(min_start, max_end + 1)], stats, weak_accept_reasons

    # === Check for subclause ===
    # We use a simple regular expression.
    clause_start = 0
    for match in re.finditer(r"([:,;]\s)|(\sand\s)|(\s[-–—]\s)|(—)", sent.text, re.IGNORECASE):

        # Check if all annoations are within the subclause
        clause_end = match.span()[0]
        clause_complete = True
        for i, ann in enumerate([value, unit, entity, property]):
            if ann == [] or (clause_start <= ann[0] and ann[1] <= clause_end):
                continue
            else:
                clause_complete = False
                break

        if clause_complete:
            candidate_annotations = []
            for start, end in annotations:
                if clause_start <= start and end <= clause_end:
                    candidate_annotations.append((start, end))

            if len(candidate_annotations) == 1:
                stats["accept_cause"]["choose_single_annotation_based_on_subclause"] += 1
                weak_accept_reasons.append("choose_single_annotation_based_on_subclause")
                return (
                    [(candidate_annotations[0][0], candidate_annotations[0][1])],
                    stats,
                    weak_accept_reasons,
                )
        elif i > 1:
            break

        clause_start = match.span()[1]

    # === Use shortest path heuristic ===
    shortest_path_annotation_idx = shortest_path(sent, annotations, value)

    if len(shortest_path_annotation_idx) == 1:
        stats["accept_cause"]["choose_single_annotation_based_on_shortest_path"] += 1
        weak_accept_reasons.append("choose_single_annotation_based_on_shortest_path")
        annotations = [annotations[shortest_path_annotation_idx[0]]]

    return annotations, stats, weak_accept_reasons


def prevent_mixing_up_length_dimensions(
    property_uri: str,
    context_annotations: Annotations,
    value_annotations: list[Offset],
    sent: Span,
    doc: Doc,
    nlp: Language,
    LABELS: Labels,
    stats: dict[str, defaultdict[int]],
    debug_mode: bool = False,
):
    """Prevent mixing up width, height and depth (e.g., "height
    of 123 ft, 🍊width🍊 of 45 m and depth of 🍏56🍏 🍓m🍓")
    """

    discard = False

    # A selection of subproperties of length
    # Wikidata-Query:
    #   SELECT DISTINCT ?subProperties ?subPropertiesLabel WHERE {
    #      ?subProperties wdt:P1647* wd:P2043.
    #      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    #    }
    width_heigt_depth = {
        "http://www.wikidata.org/entity/P2049": "width",
        "http://www.wikidata.org/entity/P2048": "height",
        "http://www.wikidata.org/entity/P4511": "depth",  # vertical depth
        "http://www.wikidata.org/entity/P5524": "depth",  # horizontal depth
        "http://www.wikidata.org/entity/P2610": "thickness",
    }

    if property_uri in width_heigt_depth.keys():
        width_heigt_depth_aliases = []
        property_dim = width_heigt_depth[property_uri]
        for uri, dim in width_heigt_depth.items():
            if dim != property_dim:
                prop_aliases = LABELS["properties"].get(uri)
                if prop_aliases is not None:
                    width_heigt_depth_aliases += prop_aliases

        width_heigt_depth_matcher = PhraseMatcher(
            nlp.vocab, attr="LEMMA", validate=debug_mode
        )

        with nlp.select_pipes(
            disable=["merge_entities", "ner", "custom_sentencizer", "coreferee"]
        ):
            dims = list(nlp.pipe(width_heigt_depth_aliases))
            width_heigt_depth_matcher.add("WIDTH_HEIGHT_DEPTH", dims)

        # Get split with value annotation inside 
        # TODO: also split on 'and'
        splits = [s + ", " for s in sent.text.split(", ") if s]
        splits[-1] = splits[-1][:-2]  # remove ", " from last split
        end_offsets = [0] + list(itertools.accumulate([len(split) for split in splits]))
        split_idx = bisect_right(
            end_offsets, value_annotations[0][0]
        )
        split_start = end_offsets[split_idx - 1] + doc[sent.start].idx
        split_end = end_offsets[split_idx] + doc[sent.start].idx
        relevant_split = doc.char_span(split_start, split_end, alignment_mode="expand")

        # Check for width, depth, height
        matches, stats, _, _ = apply_spacy_matcher(
            width_heigt_depth_matcher,
            relevant_split,
            doc,
            nlp,
            stats,
            [],
            expand_to_amods=False,
            expand_to_noun_chunk=False,
            expand_to_ner_annotation=False,
        )

        if len(matches) > 0:
            if len(matches["WIDTH_HEIGHT_DEPTH"]) == 1 and (
                matches["WIDTH_HEIGHT_DEPTH"][0][0]
                >= context_annotations["PROPERTY"][0][0]
                and matches["WIDTH_HEIGHT_DEPTH"][0][1]
                <= context_annotations["PROPERTY"][0][1]
            ):
                pass
            else:
                stats["dropped_cause"]["prevent_mixing_up_height_width_and_depth"] += 1
                discard = True
    return discard, stats


def filter_context_annotations(
    context_annotations: Annotations,
    value_annotations: list[Offset],
    unit_annotations: list[Offset],
    sent: Span,
    doc: Doc,
    nlp: Language,
    property_uri: str,
    LABELS: Labels,
    stats: dict[str, defaultdict[int]],
    value: Decimal,
    is_count: bool,
    deviations: list[Union[float, int]],
    weak_accept_reasons: list[str],
    debug_mode: bool = False,
) -> tuple[Annotations, bool]:
    """At least one measured entity and one measured property must match.
    For multiple matches special filter rules may apply.
    Additionally, neither the measured entity nor the measured property
    are allowed to be preceeded by a string like "of which".
    Finally, mixing up width and depth is prevented.
    """

    discard = False

    ents = context_annotations.get("ENTITY")
    if ents == None:
        # Could not align the measured entity,
        # thus proceed with the next sentence.
        stats["dropped_cause"]["no_measured_entity"] += 1
        discard = True

    props = context_annotations.get("PROPERTY")
    if (not discard) and (props == None):
        # Could not align the measured property
        nbr_digits = len(value.as_tuple())
        if (deviations[0] == 0) and (
            (
                is_count == True
                and nbr_digits > 4
                and float(value) not in [10_000, 100_000, 1_000_000, 10_000_000]
            )
            or (
                is_count == False
                and nbr_digits > 2
                and not any(
                    sent.text[value_annotations[0][1] :]
                    .lstrip(" ")
                    .startswith(percent_alias)
                    for percent_alias in ["%", "percent", "pct."]
                )
            )
        ):
            # Accept anyway.
            weak_accept_reasons.append("no_property_but_rare_num_exact_match")
            context_annotations["PROPERTY"] = []
            pass
        else:
            # Proceed with the next sentence.
            stats["dropped_cause"]["no_measured_property"] += 1
            discard = True

    if not discard:

        unit = [] if unit_annotations["UNIT"] == [] else unit_annotations["UNIT"][0]
        props = [] if props == None else context_annotations["PROPERTY"]

        if len(props) > 1:
            # Got too many property matches.
            (
                context_annotations["PROPERTY"],
                stats,
                weak_accept_reasons,
            ) = reduce_ambiguity(
                sent,
                context_annotations["PROPERTY"],
                stats,
                weak_accept_reasons,
                value_annotations[0],
                unit,
                entity=ents[0] if len(ents) == 1 else [],
            )
            if len(context_annotations["PROPERTY"]) > 1:
                stats["dropped_cause"]["multiple_properties"] += 1
                discard = True

        if (not discard) and (len(ents) > 1):
            # Got too many entity matches.
            (
                context_annotations["ENTITY"],
                stats,
                weak_accept_reasons,
            ) = reduce_ambiguity(
                sent,
                context_annotations["ENTITY"],
                stats,
                weak_accept_reasons,
                value_annotations[0],
                unit,
                property=props[0] if len(props) == 1 else [],
            )
            if len(context_annotations["ENTITY"]) > 1:
                stats["dropped_cause"]["multiple_entities"] += 1
                discard = True

    if not discard:
        discard = filter_shares(sent, context_annotations["ENTITY"][0])
        if discard:
            stats["dropped_cause"]["of_which_before_entity_annotation"] += 1

    if not discard and len(props) == 1:
        discard = filter_shares(sent, context_annotations["PROPERTY"][0])
        if discard:
            stats["dropped_cause"]["of_which_before_property_annotation"] += 1

    if not discard and len(props) > 0:
        discard, stats = prevent_mixing_up_length_dimensions(
            property_uri,
            context_annotations,
            value_annotations,
            sent,
            doc,
            nlp,
            LABELS,
            stats,
            debug_mode,
        )

    return context_annotations, stats, discard, weak_accept_reasons


def filter_examples_from_page(
    page_examples: defaultdict[int, list[dict]],
    stats: dict[str, defaultdict[int]],
) -> list[dict]:
    """Filter all examples created for a page. Sometimes different facts lead to
    contradictory annoations for a sentence or the same training example is created
    multiple times, because there are multiple similar coexisting statements about
    a property (e.g., the area of Albama is stated to be 135_765, 134_000±500 or
    131_365±0.5 square kilometres). Comparing a pair of examples for the same
    sentence at a time, the following filter rules are applied:

        * If everthing matches just take one of them
        * Drop example with less qualifier annotations
        * If both have qualfiers defined which differ, drop both.
        * If same value but different entity, property and or unit, discard both.
        * Exclude contradictory statements.
    """

    filtered_page_examples = []
    for sent_examples in page_examples.values():
        dropped = []
        if len(sent_examples) > 1:
            for i, ex_i in enumerate(sent_examples):
                for j, ex_j in enumerate(sent_examples):
                    if i != j and not (i in dropped):

                        e_match = ex_i["annotations"]["entity"] == ex_j["annotations"]["entity"]
                        ex_i_prop = ex_i["annotations"].get("property", [])
                        ex_j_prop = ex_j["annotations"].get("property", [])
                        p_match = ex_i_prop == ex_j_prop
                        v_match = ex_i["annotations"]["value"] == ex_j["annotations"]["value"]
                        u_match = ex_i["annotations"].get("unit", []) == ex_j["annotations"].get("unit", [])

                        # Qualifiers
                        q0_match = ex_i["annotations"].get("point_in_time", []) == ex_j["annotations"].get("point_in_time", [])
                        q1_match = ex_i["annotations"].get("start_time", []) == ex_j["annotations"].get("start_time", [])
                        q2_match = ex_i["annotations"].get("end_time", []) == ex_j["annotations"].get("end_time", [])
                        q3_match = ex_i["annotations"].get("location", []) == ex_j["annotations"].get("location", [])
                        q4_match = ex_i["annotations"].get("applies_to_part", []) == ex_j["annotations"].get("applies_to_part", [])
                        q5_match = ex_i["annotations"].get("in_scope_of", []) == ex_j["annotations"].get("in_scope_of", [])
                        q6_match = ex_i["annotations"].get("criterion_used", []) == ex_j["annotations"].get("criterion_used", [])
                        q7_match = ex_i["annotations"].get("determination_method", []) == ex_j["annotations"].get("determination_method", [])
                        q8_match = ex_i["annotations"].get("according_to", []) == ex_j["annotations"].get("according_to", [])
                        q9_match = ex_i["annotations"].get("some_qualifier", []) == ex_j["annotations"].get("some_qualifier", [])
                        q_match = q0_match and q1_match and q2_match and q3_match and q4_match and q5_match and q6_match and q7_match and q8_match and q9_match

                        # Number of annotated spans per example
                        ex_i_ann_count = sum(len(ann) for ann in ex_i["annotations"].values())
                        ex_j_ann_count = sum(len(ann) for ann in ex_j["annotations"].values())

                        if e_match and p_match and v_match and u_match:
                            if q_match:
                                # If everthing matches just take one of them
                                # (it has to be the one which might already be dropped, that is, ex_j)
                                stats["dropped_cause"]["same_example_already_exists"] += 1
                                dropped.append(j)
                            elif ex_i_ann_count < ex_j_ann_count:
                                # Drop example with less qualifier annotations
                                stats["dropped_cause"]["same_example_with_more_qualifiers_already_exists"] += 1
                                dropped.append(i)
                            elif ex_j_ann_count < ex_i_ann_count:
                                # Drop example with less qualifier annotations
                                stats["dropped_cause"]["same_example_with_more_qualifiers_already_exists"] += 1
                                dropped.append(j)
                            else:
                                # If both have qualfiers defined which differ, drop both.
                                stats["dropped_cause"]["same_example_with_as_many_but_different_qualifiers_already_exists"] += 2
                                dropped += [i, j]
                        elif not p_match and (e_match and v_match and u_match):
                            if ex_i_prop == []:
                                stats["dropped_cause"]["same_example_with_property_match_exists"] += 1
                                dropped.append(i)
                            elif ex_j_prop == []:
                                stats["dropped_cause"]["same_example_with_property_match_exists"] += 1
                                dropped.append(j)
                            else:
                                stats["dropped_cause"]["same_example_with_different_property_exists"] += 2
                                dropped += [i, j]

                        elif v_match and (not e_match or not u_match):
                            # If same value but different entity, property and or unit, discard both.
                            stats["dropped_cause"]["same_example_with_different_entity_or_unit_exists"] += 2
                            dropped += [i, j]

                        elif not v_match and (e_match and p_match and u_match and q_match):
                            # Exclude contradictory statements.
                            # (Do not remove u_match, since otherwise quantities from convert
                            #  template like "470,000 km2 (181,000 sq mi)" would be excluded.)
                            stats["dropped_cause"]["same_example_with_different_value_exists"] += 2
                            dropped += [i, j]

        dropped = list(set(dropped))
        filtered_page_examples += [ex for i, ex in enumerate(sent_examples) if i not in dropped]

    return filtered_page_examples, stats
