import os
import json
import logging
import datetime
import urllib.parse
from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict
from wikimeasurements.utils.general_utils import init_logger
from wikimeasurements.utils.wikidata_utils import (
    get_labels,
    write_to_json,
    get_page_mappings,
    shorten_using_prefix,
    get_unit_conversion_data,
    get_quantitative_statements,
    get_all_quantitative_properties,
    remove_stopwords_and_duplicates,
)


logger = init_logger("logs/wikidata.log", logging.DEBUG)

def get_distinct_resources(results: list[dict[dict]]):
    """Get distinct URIs from results in the form of results["results"]["bindings"]

    :param results: List of result dicts (e.g.,
                    [{'entity': {'type': 'uri', 'value': 'http://www.wikidata....ity/Q14381'},
                    'article': {...}, 'value': {...}, 'unit': {...}, 'property': {...}}, ...])
    :type results: list
    :return: List of distinct entities, units, qualifiers, qualifier values and qualifier units
    :rtype: list
    """

    entities = []
    units = []
    qualifiers = []
    qualifier_values = []
    qualifier_units = []

    for row in results:
        entities.append(row["entity"]["value"])
        units.append(row["unit"]["value"])

        if row.get("qualifiers") != [""]:

            qualifiers += row["qualifiers"]

            # Using group_concat in the query we always have
            # a literal. However, if a literal starts with
            # "htttp://" we can assume that it is a URI.
            for qv in row["qualifier_values"]:
                if (qv != None) and (qv.startswith("http://")):
                    qualifier_values.append(qv)

        if row.get("qualifier_units") != [""]:
            qualifier_units += row["qualifier_units"]

    entities = shorten_using_prefix(set(entities))
    units = shorten_using_prefix(set(units))
    qualifiers = shorten_using_prefix(set(qualifiers))
    qualifier_values = shorten_using_prefix(set(qualifier_values))
    qualifier_units = shorten_using_prefix(set(qualifier_units))

    return entities, units, qualifiers, qualifier_values, qualifier_units


def get_labels_for_quant_statements(
    entities,
    units,
    qualifiers,
    qual_values,
    qual_units,
    cache_dir="./cached_data/",
    use_cache: bool = True,
):
    """Get labels and aliases for all URIs in the result set of quantitative statements"""

    # Get labels from cache
    label_cache = os.path.join(cache_dir, "label_cache.json")

    if (
        use_cache
        and os.path.isfile(label_cache)
        and os.path.getsize(label_cache) > 0
    ):
        with open(label_cache, encoding="utf-8") as f:
            cached_labels = json.load(f)

        cached_entities = set(shorten_using_prefix(cached_labels["entities"].keys()))
        cached_units = set(shorten_using_prefix(cached_labels["units"].keys()))
        cached_qualifiers = set(shorten_using_prefix(cached_labels["qualifiers"].keys()))
        cached_qvalues = set(shorten_using_prefix(cached_labels["qualifiers_values"].keys()))
        cached_qunits = set(shorten_using_prefix(cached_labels["qualifiers_units"].keys()))

        # Since the entity lists are very long, cached_entities should be a set,
        # because sets appear to have faster lookpus than lists.
        new_entities = [e for e in entities if e not in cached_entities]
        new_units = [u for u in units if u not in cached_units]
        new_qualifiers = [q for q in qualifiers if q not in cached_qualifiers]
        new_qvalues = [qv for qv in qual_values if qv not in cached_qvalues]
        new_qunits = [qu for qu in qual_units if qu not in cached_qunits]

        labels = cached_labels
    else:

        new_entities = entities
        new_units = units
        new_qualifiers = qualifiers
        new_qvalues = qual_values
        new_qunits = qual_units

        labels = {
            "entities": {},
            "units": {},
            "qualifiers": {},
            "qualifiers_values": {},
            "qualifiers_units": {},
        }

    # Query all labels and aliases for distinct URIs
    logger.info("Query Wikidata for labels and aliases...")

    pbar_descr = "Processing entities"
    entity_labels = get_labels(
        new_entities, discard_abbr=True, pbar_descr=pbar_descr
    )
    labels["entities"].update(entity_labels)
    labels["entities"] = remove_stopwords_and_duplicates(labels["entities"])

    pbar_descr = "Processing units"
    unit_labels = get_labels(new_units, pbar_descr=pbar_descr)
    labels["units"].update(unit_labels)
    labels["units"] = remove_stopwords_and_duplicates(labels["units"])

    pbar_descr = "Processing qualifiers"
    new_qualifiers_wd = [q.replace("pq:", "wd:") for q in new_qualifiers]
    qualifiers_labels = get_labels(new_qualifiers_wd, pbar_descr=pbar_descr)
    labels["qualifiers"].update(qualifiers_labels)
    labels["qualifiers"] = remove_stopwords_and_duplicates(labels["qualifiers"])

    pbar_descr = "Processing qualifier values"
    qvalue_labels = get_labels(new_qvalues, pbar_descr=pbar_descr)
    labels["qualifiers_values"].update(qvalue_labels)
    labels["qualifiers_values"] = remove_stopwords_and_duplicates(
        labels["qualifiers_values"]
    )

    pbar_descr = "Processing qualifier units"
    qunit_labels = get_labels(new_qunits, pbar_descr=pbar_descr)
    labels["qualifiers_units"].update(qunit_labels)
    labels["qualifiers_units"] = remove_stopwords_and_duplicates(
        labels["qualifiers_units"]
    )

    write_to_json(labels, label_cache)

    return labels


def rearrange_results(quantitative_statements, page_mappings):
    """Re-arrange results for quantitative statements obtained from Wikidata
    and map article URLs to ID."""

    facts = defaultdict(list)

    with tqdm(quantitative_statements) as pbar:

        pbar.set_description(
            "Re-arrange Wikidata results and map article URLs to ID"
        )
        missing_mappings = []

        for row in pbar:

            # TODO: Are datatypes other than decimal used for value?
            entity = row.get("entity", {}).get("value")
            property = row.get("property", {}).get("value")
            value = row.get("value", {}).get("value")
            value_lowerbound = row.get("lowerbound", {}).get("value")
            value_upperbound = row.get("upperbound ", {}).get("value")
            unit = row.get("unit", {}).get("value")
            qualifier = row.get("qualifiers")
            qualifier_value = row.get("qualifier_values")
            qualifier_unit = row.get("qualifier_units")
            qualifier_lowerbound = row.get("qualifier_lowerbounds")
            qualifier_upperbound = row.get("qualifier_upperbounds")
            qualifier_time_prec = row.get("qualifier_time_precisions")
            page_url_raw = row.get("article", {}).get("value")

            # Replace %xx escaped special chars in url with their
            # single-character equivalent
            page_url = urllib.parse.unquote(page_url_raw)
            page_id = page_mappings["url_to_id"].get(page_url)

            if page_id == None:
                # If there is no associated ID, the article probably does not exist.
                # Thus, skip this quantitative fact.
                # TODO: Check that redirects are no longer being dropped here.
                missing_mappings.append(page_url)
                continue

            # TODO: Also drop units, etc. for which no label exists

            fact = {
                "entity": entity,
                "property": property,
                "value": value,
                "value_lowerbound": value_lowerbound,
                "value_upperbound": value_upperbound,
                "unit": unit,
                "qualifiers": qualifier,
                "qualifier_values": qualifier_value,
                "qualifier_lowerbounds": qualifier_lowerbound,
                "qualifier_upperbounds": qualifier_upperbound,
                "qualifier_units": qualifier_unit,
                "qualifier_time_precisions": qualifier_time_prec,
                "article": page_url,
            }

            facts[int(page_id)].append(fact)

    missing_mapping_list_str = "\n\t* " + "\n\t* ".join(set(missing_mappings))
    logger.warning(
        "Could not find a mapping from the page URL to the corresponding page ID for the following URLs:"
        + missing_mapping_list_str
    )

    return facts


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--outfile",
        default="./workflow/intermediates/wikidata_quantitative_statements.json",
        help="""Path of output file (e.g., './wikidata_quantitative_statements.json')""",
    )
    parser.add_argument(
        "--cache_dir",
        default="./workflow/intermediates/cache/",
        help="""Path to folder in which data is cached.""",
    )
    parser.add_argument(
        "--wiki_lang",
        default="enwiki",
        help="""Choose the Wikipedia to consider by language (e.g., "en" or "enwiki" for the English Wikipedia).""",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="""Ignore cache and run from scratch.""",
    )
    parser.add_argument(
        "--output_mode",
        default="production",
        help="""Choose between "debug" and "production". """,
    )
    args = parser.parse_args()

    args.wiki_lang = args.wiki_lang.removesuffix("wiki")

    properties = get_all_quantitative_properties(cache_dir=args.cache_dir,force=args.force)

    if args.output_mode == "debug":
        # Reduce set of properties for faster debugging
        properties = {
            "http://www.wikidata.org/entity/P2595": properties[
                "http://www.wikidata.org/entity/P2595"
            ],
            "http://www.wikidata.org/entity/P3086": properties[
                "http://www.wikidata.org/entity/P3086"
            ],
            "http://www.wikidata.org/entity/P2109": properties[
                "http://www.wikidata.org/entity/P2109"
            ],
            "http://www.wikidata.org/entity/P2386": properties[
                "http://www.wikidata.org/entity/P2386"
            ],
        }

    quantitative_statements = get_quantitative_statements(
        properties, wiki_lang=args.wiki_lang, cache_dir=args.cache_dir, force=args.force
    )

    # Get mappings between page id, url and title
    page_urls = list(
        set([page["article"]["value"] for page in quantitative_statements])
    )
    page_mappings = get_page_mappings(
        page_urls, wiki_lang=args.wiki_lang, cache_dir=args.cache_dir
    )

    # Get all distinct URIs
    (
        distinct_entities,
        distinct_units,
        distinct_qualifiers,
        distinct_qual_values,
        distinct_qual_units,
    ) = get_distinct_resources(quantitative_statements)

    # Get unit conversions
    # Quantities can be expressed with different units.
    # Thus, add alternative units.
    all_distinct_units = distinct_units + distinct_qual_units
    unit_conversion_data = get_unit_conversion_data(all_distinct_units)

    # Update list of all distinct units
    conv_units = []
    for u_conv_data in unit_conversion_data.values():
        if len(u_conv_data) > 0:
            for conv in u_conv_data:
                conv_units.append("wd:" + conv["to_unit"])

    all_distinct_units = list(set(all_distinct_units + conv_units))

    # Get labels and aliases
    label_mapping = get_labels_for_quant_statements(
        distinct_entities,
        all_distinct_units,
        distinct_qualifiers,
        distinct_qual_values,
        distinct_qual_units,
        cache_dir=args.cache_dir,
    )
    label_mapping.update({"properties": properties})

    # Re-arrange Wikidata results and map article URLs to ID
    facts = rearrange_results(quantitative_statements, page_mappings)

    metadata = {
        "title": "Quantitative Statements from Wikidata",
        "description": "The keys are the respective Wikipdia page IDs.",
        "created_at": str(datetime.datetime.now()),
    }
    output = {
        "metadata": metadata,
        "labels": label_mapping,
        "quantitative_statements": facts,
        "unit_conversion_data": unit_conversion_data,
    }

    # Write intermediate dataset of quantitative statements and the corresponding labels to disk
    write_to_json(output, args.outfile)

    logger.info("Finished! ✔️")
