import re
from typing import Union
from collections import defaultdict, OrderedDict
from spacy.language import Language
from spacy.tokens import Token
from spacy.tokens.span import Span
from spacy.tokens.doc import Doc
from spacy.symbols import (
    VERB,
    NOUN,
    ADJ,
    PROPN,
    amod,
    advmod,
    DATE,
    ORDINAL,
)
from spacy.matcher import PhraseMatcher
import networkx as nx
from wikimeasurements.utils.type_aliases import Offset, Annotations
from wikimeasurements.utils.quantity_utils import (
    get_number_spellings,
    get_digit_notations,
    get_number_patterns,
)


NUMBER_PATTERN, _, ORDINAL_PATTERN = get_number_patterns()

def get_span_distance(a: Offset, b: Offset) -> int:
    span_dist = [-1, 1][a[0] - b[1] < 0] * max(a[0] - b[1], b[0] - a[1])
    return span_dist


def get_span_distances(offsets_a: list[Offset], offsets_b: list[Offset]) -> list[int]:
    assert len(offsets_a) == len(offsets_b)
    span_distances = []
    for a, b in zip(offsets_a, offsets_b):
        span_distances.append(get_span_distance(a, b))

    return span_distances


def adapt_offset(annotations: list, offset: int) -> list[Offset]:
    """Add an offset to annotations in the form of
    a list of char offsets (e.g., [(1,3),(6,7),]"""
    return [(start + offset, end + offset) for (start, end) in annotations]


def token2char_offsets(doc: Doc, token_start: int, token_end: int) -> Offset:
    """Convert the start and end offsets of a span from
    a document from token indices to char indices.

    Assumes: token_start < len(doc)
    """
    char_start = doc[token_start].idx
    char_end = char_start + len(doc[token_start:token_end].text)
    return (char_start, char_end)


def shortest_path(sent: Span, annotations: list[Offset], target: Offset):
    """Get shortest path between a target span and a list of annotation spans
    based on the dependency parse tree from spaCy.
    """

    # Get edges for networkx graph.
    edges = []
    sent_doc = sent.as_doc()
    for token in sent_doc:
        for child in token.children:
            edges.append((token.i, child.i))

    # Create a networkx graph of the dependency parse.
    dep_graph = nx.Graph(edges)

    # Loop over tokens of target and source spans
    # and calculate shortest path.
    target_span = sent_doc.char_span(target[0], target[1], alignment_mode="expand")
    path_lengths = []
    for ann in annotations:
        ann_span = sent_doc.char_span(ann[0], ann[1], alignment_mode="expand")
        shortest_paths = []
        for target_token in target_span:
            for ann_token in ann_span:
                shortest_paths.append(
                    nx.shortest_path_length(
                        dep_graph, source=ann_token.i, target=target_token.i
                    )
                )

        if len(shortest_paths) > 0:
            path_lengths.append(min(shortest_paths))

    # Get indices of annotations with shortest path to target
    shortest_path_annotation_idx = [
        i for i, item in enumerate(path_lengths) if item == min(path_lengths)
    ]

    return shortest_path_annotation_idx


def match_adjecent_noun_as_unit(
    value_annotations: list[Offset],
    sent: Span,
    doc: Doc,
    frequent_units_matcher: PhraseMatcher,
    stats: dict[str, defaultdict[int]],
) -> tuple[Annotations, bool]:
    """For counts check take nouns succeeding the value as
    unit, also check for patterns like Noun "per" Noun
    (e.g., in XY children per woman)
    """
    discard = False
    unit_offsets = []
    value_start_char = value_annotations[0][0]
    value_end_char = value_annotations[0][1]
    value_span = sent.char_span(value_start_char, value_end_char)
    if (value_span != None) and (len(value_span) > 0):
        num_token_idx = value_span[-1].i

        if (
            (doc[num_token_idx].dep_ == "nummod")
            and (doc[num_token_idx].head.pos == NOUN)  # unit must be a noun
            and (
                doc[num_token_idx].head.i > num_token_idx
            )  # assume unit to be on right side of number
        ):
            # (nummod is not in spacy.symbols)
            head_noun = doc[num_token_idx].head
            head_noun_as_span = doc[head_noun.i : head_noun.i + 1]
            is_freq_unit = len(frequent_units_matcher(head_noun_as_span)) != 0
            if is_freq_unit:
                # Unit for count can't be a popular SI unit
                stats["dropped_cause"]["unit_for_count_is_frequent_si_unit"] += 1
                discard = True
            else:
                # Get start of unit span
                tokens_inbetween = list(doc[num_token_idx : head_noun.i])
                if len(tokens_inbetween) < 10:  # just a loose heuristic
                    tokens_inbetween.reverse
                    unit_start = False
                    for t in tokens_inbetween:
                        if (t.head.i == t.i + 1) and (
                            doc[t.i].dep_
                            in [
                                "amod",
                                "compound",
                            ]
                        ):
                            continue
                        elif t.i == num_token_idx:
                            # Token next to number is start (e.g, in "1.61 buildings")
                            unit_start = num_token_idx + 1
                        elif (t.i == num_token_idx + 1) and (t.text in ["(", "-"]):
                            # Two tokens next to number is start
                            # (e.g, in "1.61 (Pauling scale)" or "120-seat")
                            unit_start = num_token_idx + 2
                        else:
                            # Couldn't find appropriate start
                            break

                    # Get end of unit span
                    if unit_start:
                        nbr_tokens = len(doc)
                        if (
                            (head_noun.i + 2 < nbr_tokens)
                            and (doc[head_noun.i + 1].text == "per")
                            and (doc[head_noun.i + 2].pos == NOUN)
                        ):
                            # Identify units like "children per woman" in sentences such as 
                            # "[...] the fertility rate across COUNTRY was NUM children per woman."
                            unit_end = head_noun.i + 3
                        elif (
                            (head_noun.i + 3 < nbr_tokens)
                            and (doc[head_noun.i + 1].pos == VERB)
                            and (doc[head_noun.i + 1].dep_ == "acl")
                            and (doc[head_noun.i + 1].head.i == head_noun.i)
                            and (doc[head_noun.i + 2].text == "per")
                            and (doc[head_noun.i + 3].pos == NOUN)
                        ):
                            # Identify units like "children born per woman" in sentences such as
                            # "[...] the fertility rate of COUNTRY is NUM children born per woman."
                            unit_end = head_noun.i + 4
                        else:
                            # Identify units like "homes" in sentences such as
                            # "[...] enough energy to power 1,000,000 homes [...]"
                            unit_end = head_noun.i + 1

                        unit_offsets.append(
                            token2char_offsets(doc, unit_start, unit_end)
                        )
                        unit_offsets = adapt_offset(unit_offsets, -sent.start_char)

    return {"UNIT": unit_offsets}, stats, discard


def apply_coreference_resolution(
    sent: Span,
    coref_context: list[Span],
    doc: Doc,
    nlp: Language,
    coref_cache: dict[Span, tuple[Union[Doc, None], Union[int, None]]],
    at_most_sents_before: int = 1,
):
    """Apply coreference resolution to sentence and its preceding text."""
    coref_from_cache = coref_cache.get(sent)
    if coref_from_cache is None:
        # Consider the sentences before until the previous
        # paragraph but at most x sentences before.
        sent_window = coref_context[-(at_most_sents_before + 1) :]

        if len(sent_window) > 1:
            with nlp.select_pipes(
                disable=["merge_entities", "ner", "custom_sentencizer"]
            ):
                coref_offset = sent_window[0].start
                coref_doc = nlp(doc[coref_offset : sent_window[-1].end].as_doc())
        else:
            # Skip coreference resolution if sentence is first sentence of paragraph.
            coref_offset = None
            coref_doc = None

        coref_cache.update({sent: (coref_doc, coref_offset)})
    else:
        coref_doc = coref_from_cache[0]
        coref_offset = coref_from_cache[1]

    return coref_doc, coref_offset, coref_cache


def analyze_coreferee_results(
    matcher: PhraseMatcher,
    target_concepts: list,
    coref_doc: Doc,
    coref_offset: int,
    sent: Span,
    doc: Doc,
    nlp: Language,
):
    """Apply spaCy PhraseMatcher to the head spans of coreferences
    found by coreferee in the sentence of interest.
    """
    coref_matches = []
    coref_matches_tags = []
    for coref_cluster in coref_doc._.coref_chains:
        for coref in coref_cluster:
            for coref_token_i in coref:
                coref_span_start = coref_token_i + coref_offset
                if sent.start <= coref_span_start < sent.end:
                    coref_head = coref_cluster[
                        coref_cluster.most_specific_mention_index
                    ]
                    for head_token_i in coref_head:
                        # Seems like coreferee only consideres single tokens
                        head_span_start = head_token_i + coref_offset
                        head_span_end = head_span_start + 1
                        head_span = doc[head_span_start:head_span_end]
                        new_coref_matches = matcher(head_span)
                        for coref_tag_id, _, _ in new_coref_matches:
                            coref_tag = nlp.vocab.strings[coref_tag_id]
                            if coref_tag in target_concepts:
                                # Found coreference of missing concept type.
                                # Just take the first occurance of the coreference.
                                coref_matches.append(
                                    (
                                        coref_tag_id,
                                        coref_span_start,
                                        coref_span_start + 1,
                                    )
                                )
                                coref_matches_tags.append(coref_tag)

                                if all(
                                    concept in coref_matches_tags
                                    for concept in target_concepts
                                ):
                                    # Got everything, thus return early.
                                    return coref_matches, coref_matches_tags

    return coref_matches, coref_matches_tags


def analyze_xx_coref_results(
    matcher: PhraseMatcher,
    target_concepts: list,
    coref_doc: Doc,
    coref_offset: int,
    sent: Span,
    doc: Doc,
    nlp: Language,
):
    """Apply spaCy PhraseMatcher to the head spans of coreferences
    found by xx_coref in the sentence of interest.
    """
    coref_matches = []
    coref_matches_tags = []
    for coref_cluster in coref_doc._.coref_clusters:
        corefs_in_sent = []
        for coref_span in coref_cluster:
            coref_span_start = coref_span[0] + coref_offset
            coref_span_end = coref_span[1] + coref_offset
            if sent.start <= coref_span_start and coref_span_end <= sent.end:
                corefs_in_sent.append((coref_span_start, coref_span_end))

        if len(corefs_in_sent) > 0:
            # Found coreference in current sentence
            for _, head_span_offsets in coref_doc._.cluster_heads.items():
                if head_span_offsets in coref_cluster:
                    head_span = coref_doc[
                        head_span_offsets[0] : head_span_offsets[1] + 1
                    ]
                    if ", which " in head_span.text:
                        # For example, in "The state debt peaked in 1994, when it reached US$14.4 billion."
                        # (https://en.wikipedia.org/wiki/Economy_of_Bulgaria)
                        # 'it' is referencing to a "state debt, which represented 180% of the GDP"
                        # leading to 'US$14.4 billion' mistakenly associated with 'GDP'.
                        start_char = head_span[0].idx
                        new_end_char = start_char + len(
                            head_span.text.split(", which")[0]
                        )
                        head_span = coref_doc.char_span(
                            start_char,
                            new_end_char,
                            alignment_mode="expand",
                        )

                    new_coref_matches = matcher(head_span)
                    for coref_tag_id, _, _ in new_coref_matches:
                        coref_tag = nlp.vocab.strings[coref_tag_id]
                        if coref_tag in target_concepts:
                            # Found coreference of missing concept type.
                            # Just take the first occurance of the coreference.
                            coref_matches.append(
                                (
                                    coref_tag_id,
                                    corefs_in_sent[0][0],
                                    corefs_in_sent[0][1],
                                )
                            )
                            coref_matches_tags.append(coref_tag)

                            if all(
                                concept in coref_matches_tags
                                for concept in target_concepts
                            ):
                                # Got everything, thus return early.
                                return coref_matches, coref_matches_tags

                    break  # Found head

    return coref_matches, coref_matches_tags


def analyze_coreference_resolution_results(
    matcher: PhraseMatcher,
    target_concepts: list,
    coref_doc: Doc,
    coref_offset: int,
    sent: Span,
    doc: Doc,
    nlp: Language,
    available_components: list,
):
    if "coreferee" in available_components:
        coref_matches, coref_matches_tags = analyze_coreferee_results(
            matcher,
            target_concepts,
            coref_doc,
            coref_offset,
            sent,
            doc,
            nlp,
        )
    elif "xx_coref" in available_components:
        coref_matches, coref_matches_tags = analyze_xx_coref_results(
            matcher,
            target_concepts,
            coref_doc,
            coref_offset,
            sent,
            doc,
            nlp,
        )
    else:
        coref_matches, coref_matches_tags = []

    return coref_matches, coref_matches_tags


def apply_spacy_matcher(
    matcher: PhraseMatcher,
    sent: Span,
    doc: Doc,
    nlp: Language,
    stats: dict[str, defaultdict[int]],
    weak_accept_reasons: list[str],
    exclude_spans: list[tuple[int, int, str]] = [],
    expand_to_noun_chunk: bool = False,
    expand_to_amods: bool = False,
    amods: defaultdict[Token, list[Token]] = {},
    frequent_units_matcher: Union[PhraseMatcher, None] = None,
    expand_to_ner_annotation: bool = False,
    expand_to_propn: bool = False,
    check_corefs: bool = False,
    coref_context: list = [],
    coref_cache: dict[Doc, int] = {},
    is_count: Union[bool, None] = None,
    deviations: list[Union[float, int]] = [],
) -> Annotations:
    """Apply the given Spacy Phrasematcher to a text span.
    Matches are returned as char offsets.
    """

    matches = matcher(sent)
    merged_annotations = defaultdict(list)
    nbr_matches = len(matches)
    tags = [nlp.vocab.strings[match_id] for match_id, _, _ in matches]

    if check_corefs:
        # greedy_but_slow_trigger = "ENTITY" not in tags or "PROPERTY" not in tags
        lazy_but_fast_trigger = (
            not is_count
            and deviations[0] == 0
            and "PROPERTY" in tags
            and "ENTITY" not in tags
        )
        if lazy_but_fast_trigger:

            available_components = nlp.pipe_names
            known_coref_components = ["coreferee", "xx_coref"]
            if not any(comp in available_components for comp in known_coref_components):
                print(
                    "Could not find a spaCy component for coreference resolution, thus omit coreference resolution."
                )
            else:
                # Check for coreferences.
                coref_doc, coref_offset, coref_cache = apply_coreference_resolution(
                    sent, coref_context, doc, nlp, coref_cache, at_most_sents_before=1
                )

                if coref_doc is not None:

                    # Collect missing tags.
                    target_concepts = []
                    for concept in ["ENTITY", "PROPERTY"]:
                        if concept not in tags:
                            target_concepts.append(concept)

                    # Check for matches in coreference heads.
                    (
                        coref_matches,
                        coref_matches_tags,
                    ) = analyze_coreference_resolution_results(
                        matcher,
                        target_concepts,
                        coref_doc,
                        coref_offset,
                        sent,
                        doc,
                        nlp,
                        available_components,
                    )

                    nbr_coref_matches = len(coref_matches)
                    if nbr_coref_matches > 0:

                        matches += coref_matches
                        nbr_matches += nbr_coref_matches
                        tags += coref_matches_tags

                        if "ENTITY" in coref_matches_tags:
                            stats["accept_cause"][
                                "found_entity_with_coreference_resolution"
                            ] += 1
                            weak_accept_reasons.append(
                                "found_entity_with_coreference_resolution"
                            )

                        if "PROPERTY" in coref_matches_tags:
                            stats["accept_cause"][
                                "found_property_with_coreference_resolution"
                            ] += 1
                            weak_accept_reasons.append(
                                "found_property_with_coreference_resolution"
                            )

                    if "ENTITY" not in tags:
                        # Entity tag is required, thus return early if there is no entity match.
                        return (
                            merged_annotations,
                            stats,
                            weak_accept_reasons,
                            coref_cache,
                        )

    if nbr_matches > 0:

        if expand_to_ner_annotation:
            measured_ents_types = [
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
                "WORK_OF_ART",
            ]
            considered_ents = [ent for ent in sent.ents if ent.label_ in measured_ents_types]

        # Merge annotations for cases like "surface area",
        # where surface, area and surface area are matched:
        #   [16488956749667594207, 88, 89]
        #   [16488956749667594207, 88, 90]
        #   [16488956749667594207, 89, 90]
        merged_matches = []
        last_idx = -1
        for (match_id, start, end) in matches:
            if last_idx >= 0:
                if match_id == last_match_id and start <= last_end:
                    merged_matches[last_idx][1] = min(start, last_start)
                    merged_matches[last_idx][2] = max(end, last_end)
                    continue

            merged_matches.append([match_id, start, end])
            last_match_id = match_id
            last_start = start
            last_end = end
            last_idx += 1

        annotations = defaultdict(list)
        last_match_end = 0
        for i, (match_id, start, end) in enumerate(merged_matches):

            tag = nlp.vocab.strings[match_id]

            if expand_to_amods or expand_to_noun_chunk or expand_to_ner_annotation:
                # For the expansion of annotations it is assumed that the matches are sorted

                next_match_start = (
                    matches[i + 1][1] if i < (nbr_matches - 1) else sent.end
                )

                if expand_to_amods:
                    for token in doc[start:end]:
                        # Get amod of amod of amod ... of annotation token.
                        # Additionally, expand to compounds.
                        head_token = token
                        break_condition = False
                        while head_token in amods.keys() and len(amods[head_token]) > 0:
                            for amod_token in amods[head_token]:

                                if (
                                    (tag == "PROPERTY")
                                    and (amod_token.pos == ADJ)
                                    and (amod_token.i > 1)
                                    and (doc[amod_token.i - 2].ent_type_ == "ORDINAL")
                                    and (doc[amod_token.i - 1].text == "-")
                                ):
                                    # Do not expand to pattern like "{ordinal}-{ADJ}" for
                                    # properties (e.g., in "🍊fourth-highest elevation🍊")
                                    break_condition = True
                                    break

                                if frequent_units_matcher is not None:
                                    # Do not expand into kwnown unit!
                                    # "The 🌶️aircraft🌶️ has a 🍏12,345🍏 🍓km🍓 / 6,666 🍊nmi range🍊."
                                    amod_as_span = doc[amod_token.i : amod_token.i + 1]
                                    is_freq_unit = (
                                        len(frequent_units_matcher(amod_as_span)) != 0
                                    )
                                    if is_freq_unit:
                                        break_condition = True
                                        break

                                # Expand annotation
                                if amod_token.i < start:
                                    start = max(amod_token.i, last_match_end)
                                elif amod_token.i > end:
                                    end = 1 + min(amod_token.i, next_match_start)

                                head_token = amod_token

                            if break_condition == True:
                                break

                if expand_to_noun_chunk:
                    # Expand annotation to full noun chunk
                    for noun_chunk in sent.noun_chunks:
                        if doc[start] in noun_chunk:
                            # Expand annotation
                            start = max(noun_chunk.start, last_match_end)
                            end = 1 + min(noun_chunk.end, next_match_start)
                            break

                if tag == "ENTITY":

                    if expand_to_propn and not any(
                        t.pos != PROPN for t in doc[start:end]
                    ):
                        # Expand PROPN to adjecent PROPNs
                        # (e.g., "🌶️Mount Pleasant🌶️ Municipal Airport"
                        #     to "🌶️Mount Pleasant Municipal Airport🌶️")

                        # Expand to left
                        tokens_before = list(doc[sent.start : start])
                        tokens_before.reverse()
                        for token in tokens_before:
                            if token.pos == PROPN:
                                start = max(token.i, last_match_end)
                            else:
                                break

                        # Expand to right
                        tokens_after = list(doc[end : sent.end])
                        for token in tokens_after:
                            if token.pos == PROPN:
                                end = 1 + min(token.i, next_match_start)
                            else:
                                break

                    if expand_to_ner_annotation:
                        # Expand annotation to NER annotations
                        # (e.g., 🌶️A340🌶️-300 is expanded to 🌶️A340-300🌶️)
                        for ent in considered_ents:
                            start_inbetween = ent.start <= start < ent.end
                            end_inbetween = ent.start < end <= ent.end
                            if start_inbetween or end_inbetween:

                                if (
                                    ent.end < len(doc)
                                    and doc[ent.end].text == "CDP"
                                    and ent.end < next_match_start
                                ):
                                    # Expand annotation for cities followed by "CDP" (Census-designated place)
                                    # (e.g., "🌶️Jülich City CDP🌶️ has a 🍊total area🍊 of 🍏35🍏 🍓square miles🍓")
                                    # I don't ensure ent.label == GPE, because the NER model makes a lot of mistakes.
                                    end = 1 + ent.end

                                if start > ent.start:
                                    start = max(min(ent.start, start), last_match_end)

                                if end < ent.end:
                                    end = 1 + min(
                                        max(ent.end - 1, end), next_match_start
                                    )
                                # Spacy annotations do not overlap, thus we are done.
                                break

                last_match_end = end

            # Convert token offsets to char offsets
            char_offsets = token2char_offsets(doc, start, end)

            # Convert document char offsets to sentence char offsets
            char_offsets = [
                char_offsets[0] - sent.start_char,
                char_offsets[1] - sent.start_char,
            ]

            # Prevent overlaps with already existing annotations
            discard = False
            for (exclude_start, exclude_end, exclude_tag) in exclude_spans:
                start_inbetween = exclude_start <= char_offsets[0] < exclude_end
                end_inbetween = exclude_start < char_offsets[1] <= exclude_end
                if start_inbetween and end_inbetween:
                    if tag == "PROPERTY" and exclude_tag == "UNIT":
                        # Property and unit spans can overlap (for example in "🍏13🍏 🍊🍓episodes🍊🍓")
                        # We only check for PROPERTY tag, since units are matched before properties.
                        stats["accept_cause"]["properties_and_units_may_overlap"] += 1
                        pass
                    else:
                        # Discard annotation
                        stats["dropped_cause"]["overlapping_annotations"] += 1
                        discard = True
                        break
                elif start_inbetween:
                    whitespace = re.search(r"\S", sent.text[exclude_end:])
                    whitespace = 0 if whitespace is None else whitespace.start()
                    char_offsets[0] = exclude_end + whitespace
                elif end_inbetween:
                    whitespace = re.search(r"\S", sent.text[exclude_end::-1])
                    whitespace = 0 if whitespace is None else whitespace.start()
                    char_offsets[1] = exclude_start - whitespace

            if char_offsets[0] >= char_offsets[1]:
                # Annotations cannot be of zero or negative length
                discard = True

            if not discard:
                annotations[tag].append(tuple(char_offsets))

        for tag in annotations.keys():
            # TODO: test if this functions as expected
            merged_annotations[tag] = merge_annotation_offsets(annotations[tag])

    return merged_annotations, stats, weak_accept_reasons, coref_cache


def visualize_annotations(text: str, annotations: tuple[list[Offset], str]) -> str:
    """Visualize annotations by enclosing them with given symbols
    (e.g., "In 📆2370📆, 🍊life expectancy🍊 in 🌶️Europe🌶️ was 🍏120🍏 🍓years🍓")

    'annotations' is a list of tuples, where the first item is a
    list of char offsets and the second item is a symbol.
    For example:
     [
         ([(16, 22),(28, 35)], '🌶️'),
         ([(3, 7)], '📆'),
         ...
     ]
    """

    # Flatten list of annotations and add a label
    ann_offsets_with_label = []
    for (ann, tag) in annotations:
        ann_offsets = list(sum(ann, ()))
        ann_offsets_with_label += [(offset, tag) for offset in ann_offsets]

    # Get tag order for grouping by tag
    ann_offsets_with_label = sorted(ann_offsets_with_label, key=lambda x: x[0])
    tag_order = list(OrderedDict.fromkeys([t[1] for t in ann_offsets_with_label]))

    # Sort from large to small whilst ensuring that
    # annotations are grouped by their tag
    ann_offsets_with_label = sorted(
        ann_offsets_with_label,
        key=lambda x: (x[0], tag_order.index(x[1])),
        reverse=True,
    )

    # Annotate sentence
    text_ann = text
    for offset, label in ann_offsets_with_label:
        text_ann = text_ann[:offset] + label + text_ann[offset:]

    return text_ann


def inline_tags_to_char_offsets(example: str) -> tuple[str, Annotations]:
    """Calculate char offsets based on enclosing symbols for examples like: 
    "🌶️Jülich🌶️ has a 🍊surface area🍊 of 🍏90,39🍏 🍓square kilometres🍓"
    """
    regex = r"(🍊|🌶️|🍏|🍓|📆)"
    sentence = re.sub(regex, "", example)
    offset = 0
    annotations = defaultdict(list)
    for match in re.finditer(regex, example):

        symbol = match.group()

        if symbol == "🌶️":
            tag = "entity"
        elif symbol == "🍊":
            tag = "property"
        elif symbol == "🍏":
            tag = "value"
        elif symbol == "🍓":
            tag = "unit"
        elif symbol == "📆":
            tag = "temporal_scope"
        else:
            raise ValueError

        annotations[tag].append(match.start() - offset)
        offset += len(symbol)

    tuple_annotations = defaultdict(list)

    for k, v in annotations.items():
        assert len(v) % 2 == 0
        for i in range(0, len(v), 2):
            tuple_annotations[k].append((v[i], v[i + 1]))

    return sentence, tuple_annotations


def find_datetimes(sent: Span, nlp: Language) -> list[Span]:
    """Returns all date spans in a sentence based on NER."""
    
    assert set(["ner", "merge_entities", "custom_sentencizer"]).issubset(nlp.pipe_names)
    sentence = nlp(sent.as_doc())
    dates = [ent for ent in sentence.ents if ent.label == DATE]

    return dates


def trim_and_adapt_offsets(text: str, offset: Offset) -> Offset:
    """Remove leading and trailing whitespace from annotations
    and adapt the char offsets accordingly.
    """
    annotation_span = text[offset[0] : offset[1]]
    leading_whitespace_count = len(annotation_span) - len(annotation_span.lstrip())
    trailing_whitespace_count = len(annotation_span) - len(annotation_span.rstrip(" ."))

    return (offset[0] + leading_whitespace_count, offset[1] - trailing_whitespace_count)


def remove_dates_and_more_from_offsets(
    offsets: list[Offset],
    sent: Doc,
    consider_ordinals: bool = True,
    WIKI_ORDINALS_LOWER: Union[list[str], None] = None,
) -> list[Offset]:
    """Filter identified number spans:
    * If a number span overlaps with a date, the number span is removed.
        (e.g., here '2015' in "September 🍏2015🍏" should not be considered a number)
    * If a number span is part of a fraction, the number span is removed.
        (e.g., '100,000' in "a homicide rate of 5 murders per 🍏100,000🍏 inhabitants"
        would be removed, since it is not a claim of 100_000 inhabitants existing, but
        about a homicide rate of 5/100_000 murders/inhabitants)
    """

    # Get char offsets of identified dates
    if consider_ordinals:
        exclude = [DATE]
    else:
        exclude = [DATE, ORDINAL]

    time_units = [
        "1 millennium",
        "1 century",
        "1 decade",
        "1 year",
        "1 month",
        "1 day",
        "1 hour",
        "1 minute",
        "1 second",
    ]
    time_units += [
        "one millennium",
        "one century",
        "one decade",
        "one year",
        "one month",
        "one day",
        "one hour",
        "one minute",
        "one second",
    ]
    time_units += [
        "a single millennium",
        "a single century",
        "a single decade",
        "a single year",
        "a single month",
        "a single day",
        "a single hour",
        "a single minute",
        "a single second",
    ]
    time_units_pl = [
        "millenniums",
        "centuries",
        "decades",
        "years",
        "months",
        "days",
        "hours",
        "minutes",
        "seconds",
    ]
    exclude_spans = []
    for ent in sent.ents:
        if ent.label in exclude:
            if ent.label == DATE and (
                ent.text in time_units or any(u in ent.text for u in time_units_pl)
            ):
                # Probably, measurement and not date (e.g., "1 second" or "3.14 millenniums")
                continue
            else:
                exclude_spans.append((ent.start_char, ent.end_char))

    # Convert document char offsets to sentence char offsets
    exclude_spans = adapt_offset(exclude_spans, -sent.start_char)

    # Remove overlapping spans
    filtered_offsets = offsets.copy()
    for start, end in offsets:
        candidate_span = sent.text[start:end]

        # Remove spans which are likely part of a fraction
        # TODO: The other way around, expand num to num per num if num followed by per
        if sent.text[:start].endswith(" per ") or (
            not consider_ordinals
            and (
                ORDINAL_PATTERN.search(candidate_span) != None
                or any(
                    candidate_span.endswith(ordinal) for ordinal in WIKI_ORDINALS_LOWER
                )
            )
        ):
            filtered_offsets.remove((start, end))
        else:
            # Remove spans overlapping with dates, etc.
            for date_start, date_end in exclude_spans:
                is_overlapping = (start <= date_start < end) or (
                    start < date_end <= end
                )
                if is_overlapping and (start, end) in filtered_offsets:
                    filtered_offsets.remove((start, end))

    return filtered_offsets


def get_representations_and_match_value(
    value_str,
    sent: Span,
    doc: Doc,
    nlp: Language,
    threshold: float = 0.03,
    debug_mode: bool = False,
):
    """Perform string-matching of different representations of a given value within a text span."""

    def prepare_value_matcher(value_str):
        # Get value representations
        # TODO: This approach matches '18' in '18 billion' ignoring the full number
        spelling_options = get_number_spellings(value_str, None, None)
        digit_notations = get_digit_notations(value_str, threshold=threshold)
        value_representations = spelling_options + digit_notations

        # Prepare value matcher
        value_matcher = PhraseMatcher(nlp.vocab, attr="LOWER", validate=debug_mode)
        value_matcher.add(
            "NUMERIC_VALUE", list(nlp.tokenizer.pipe(value_representations))
        )

        return value_matcher

    value_matcher = prepare_value_matcher(value_str)
    value_matches = value_matcher(sent)
    # TODO: What if value or context matches have multiple hits for the same concept type?
    value_annotations = [token2char_offsets(doc, start, end) for _, start, end in value_matches]

    return value_annotations


def merge_annotation_offsets(
    offsets: Union[list[Offset], list[list[int, int]]]
) -> Union[list[Offset], list[list[int, int]]]:
    """Merge overlapping annotation offsets.

    Offsets can be given as an list of lists or tuples
    (e.g., [[14, 25], [17, 26], [30, 40], ... ]
     or    [(14, 25), (17, 26), (30, 40), ... ])

    The Code was adapted from
    https://stackoverflow.com/a/43600953.
    """

    # Handle empty input
    if len(offsets) == 0:
        return offsets

    # If list items are tuples, convert them lists
    tuple_list = type(offsets[0]) == tuple
    if tuple_list:
        offsets = [list(t) for t in offsets]

    # Merge overlapping offsets
    offsets.sort(key=lambda offset: offset[0])
    merged_offsets = [offsets[0]]
    for current_offs in offsets:
        previous_offs = merged_offsets[-1]
        if current_offs[0] <= previous_offs[1]:
            # Changing previous_offs also changes merged_offsets
            # since in Python assignment statements do not copy
            # objects but create bindings.
            previous_offs[1] = max(previous_offs[1], current_offs[1])
        else:
            merged_offsets.append(current_offs)

    # Convert back to list of tuples if tuples were given
    if tuple_list:
        merged_offsets = [tuple(l) for l in merged_offsets]

    return merged_offsets


def regex_number_finder(string: str) -> list[Offset]:
    """Find numbers in text using a regular expression.
    Various number representations are supported
    (i.a., 123, 10^2, 1e-3, -1.234 10^2, etc.).

    Limitations:
     * '30th' yields just '30'
    """

    # Apply regex pattern
    matches = NUMBER_PATTERN.finditer(string)

    # Get char offsets
    numbers = [match.span() for match in matches]

    return list(set(numbers))


def spacy_number_finder(sent: Span, WIKI_ORDINAL_LEMMAS: list[int]) -> list[Offset]:
    """Identify numbers in text based on Spacy tags.

    Some information on tags related to numbers:

     Spacy provides different tags that could be used for
     detecting numbers. Strings like "one-third", "1,000", etc.
     would yield the NER tag 'CARDINAL', the detailed part-of-
     speech tag (tag_) 'CD', the simpler UPOS part-of-speech tag
     (pos_) 'NUM' and 'like_num' set to True.

     Each token in "more than 300,000" would be tagged 'CARDINAL',
     however only "300,000" would be tagged 'CD', 'NUM' and
     like_num = True. Similarly, the NER tag "QUANTITY" applies to
     both the number and unit in "13000 km". Besides "CARDINAL"
     and "QUANTITY", the NER tags "ORDINAL", "PERCENT" and "MONEY"
     are related to numbers.

     We start from the "CD" tag, since we just want to identify
     numbers and are not interested in quantity modifiers like
     "more than" and units like "meters".
    """

    # Detect all tokens tagged as cardinal numbers
    cd_blacklist = ["°"]  # sometimes these tokens are wrongly tagged with CD
    cardinals = [
        token
        for token in sent
        if ((token.tag_ == "CD" or token.like_num) and token.text not in cd_blacklist)
    ]

    def expand_number_annotation(cardinal: Span, sent: Span) -> Offset:
        """Expand from cardinal numbers to adjecent symbols
        and other cardinal numbers."""

        # Initialize token offsets
        annotation_token_offsets = [cardinal.i, cardinal.i]

        signs = ["-", "+", "±", "√"]  # ∓
        number_chars = ["/", "^", "x", "e", "E"]

        # Expand to left and right
        for increment in [-1, 1]:
            last_token_inside = cardinal
            token_offset = cardinal.i + increment
            while token_offset >= sent.start and token_offset < sent.end:
                adjecent_token = sent[token_offset - sent.start]
                if (
                    adjecent_token in cardinals
                    or adjecent_token.text in signs
                    or adjecent_token.text in number_chars
                    or adjecent_token.lemma in WIKI_ORDINAL_LEMMAS
                    or "d'ddd" in adjecent_token.shape_
                ):
                    # (We check for "d'ddd" in shape since, Spacy does not tag
                    # numbers with an apostroph as thousands seperator as number)
                    annotation_token_offsets[max(0, increment)] = token_offset
                    token_offset += increment
                    last_token_inside = adjecent_token
                else:
                    break

            # No special symbol at the start or end of a number.
            # However, a number is allowed to be preceded by a sign.
            if (last_token_inside.text in number_chars) or (
                increment == 1 and last_token_inside.text in signs
            ):
                annotation_token_offsets[max(0, increment)] -= increment

        annotation_token_offsets[1] += 1  # range like this doc[50:50+1]

        return tuple(annotation_token_offsets)

    # Loop over all cardinals and expand annotation
    numbers = []
    for cd in cardinals:
        # Get token offsets
        num_ann_offsets = expand_number_annotation(cd, sent)
        numbers.append(num_ann_offsets)

    return list(set(numbers))


def extract_values_from_span(
    sent: Span,
    doc: Doc,
    consider_ordinals: bool = True,
    WIKI_ORDINAL_LEMMAS: Union[list[int], None] = None,
    WIKI_ORDINALS_LOWER: Union[list[str], None] = None,
) -> list[Offset]:
    """Identify all numbers in the given text span."""

    # Find numbers in text using a regular expression
    matches_regex = regex_number_finder(sent.text)
    nbr_regex_matches = len(matches_regex)

    # TODO: Implement gazetteer number finder

    # Since the regular expressions do not include number words
    # like 'thousand' and 'million', we additionally search for
    # numbers in text using Spacy's tags for cardinal numbers.
    matches_spacy_token = spacy_number_finder(sent, WIKI_ORDINAL_LEMMAS)
    nbr_spacy_matches = len(matches_spacy_token)

    if (nbr_regex_matches == 0) and (nbr_spacy_matches == 0):
        return []
    else:
        # Convert Spacy token offsets to char offsets
        matches_spacy = []
        if nbr_spacy_matches > 0:
            for match in matches_spacy_token:
                matches_spacy.append(token2char_offsets(doc, *match))

            # Convert document char offsets to sentence char offsets
            sent_offset = -sent.start_char
            matches_spacy = adapt_offset(matches_spacy, sent_offset)

        # Aggregate the annotations of both approaches
        all_matches = matches_spacy + matches_regex
        merged_offsets = merge_annotation_offsets(all_matches)

        # Exclude number matches which are dates according to NER
        number_offsets = remove_dates_and_more_from_offsets(
            merged_offsets,
            sent,
            consider_ordinals=consider_ordinals,
            WIKI_ORDINALS_LOWER=WIKI_ORDINALS_LOWER,
        )

        # Force start char offset to be lower than end char offset and remove whitespace.
        filtered_number_offsets = []
        for (start, end) in number_offsets:
            if start < end:
                start, end = trim_and_adapt_offsets(sent.text, (start, end))
                filtered_number_offsets.append((start, end))

        return filtered_number_offsets


def enclosed_in_parentheses(string: str, open_char: str = "(", close_char: str = ")") -> list[Offset]:
    """Find all substrings enclosed in parentheses."""

    enclosed_in_parentheses = []
    if open_char in string:
        stack = []
        for i, char in enumerate(string):
            if char == open_char:
                stack.append(i)
            elif char == close_char:
                if len(stack) > 0:
                    enclosed_in_parentheses.append((stack.pop(), i + 1))

    return enclosed_in_parentheses


def check_for_aliases_in_parentheses(sent: Span, annotations: list[Offset]):
    """If one match is directly followed by the other matches enclosed in parentheses,
    the one match before the parentheses is choosen.

    For example:
        * 'Aachen' in "Aachen (Aachen dialect: "Oche"; French and traditional English: Aix-la-Chapelle; or "Aquisgranum"; Dutch: Aken)"
        * 'IDLH' in "The IDLH (immediately dangerous to life or health) value for antimony is 50 mg/m3."
    """

    enclosed_spans = enclosed_in_parentheses(sent.text)
    if len(enclosed_spans) > 0:
        # Assume annotations are sorted
        parentheses_exception_applies = False
        is_inside = lambda a, b: a[0] >= b[0] and a[1] <= b[1]
        for encl_span in enclosed_spans:
            dist = encl_span[0] - annotations[0][1]
            if 0 <= dist <= 2:
                parentheses_exception_applies = True
                for cont_ann in annotations[1:]:
                    if not is_inside(cont_ann, encl_span):
                        parentheses_exception_applies = False
                        break

        return parentheses_exception_applies


@Language.component("custom_sentencizer")
def custom_sentencizer(doc: Doc) -> Doc:
    # Force sentence start at double linebreak
    for i, token in enumerate(doc[:-2]):
        if token.text == "\n\n":
            doc[i + 1].is_sent_start = True

    return doc


@Language.component("lower_lemmas")
def lower_case_lemmas(doc: Doc) -> Doc:
    """Convert lemmas to lowerca1se"""
    for token in doc:
        token.lemma_ = token.lemma_.lower()

    return doc


@Language.component("some_name")
def dummy_component(doc: Doc) -> Doc:
    return doc


def get_amods_and_compounds(
    sent: Span, only_consecutive: bool = True
) -> defaultdict[Token, list[Token]]:
    """Create a dictionary of consecutive amod tokens with the head
    tokens as keys (e.g., {temperature: [highest]})
    """

    amods_and_compounds = defaultdict(list)
    for canidate_token in sent:
        if canidate_token.dep in [amod, advmod] or canidate_token.dep_ == "compound":
            # (compound is not in spacy.symbols)
            amods_and_compounds[canidate_token.head].append(canidate_token)

    if only_consecutive:
        # Filter out tokens which are not adjecent
        for (head, childs) in amods_and_compounds.items():
            childs_and_head = [head] + childs
            for token in childs:
                token_offset_in_sent = token.i - sent.start

                if (
                    token_offset_in_sent - 1 < 0
                    or sent[token_offset_in_sent - 1] not in childs_and_head
                ) and (
                    token_offset_in_sent + 1 >= len(sent)
                    or sent[token_offset_in_sent + 1] not in childs_and_head
                ):
                    amods_and_compounds[head].remove(token)

    return amods_and_compounds
