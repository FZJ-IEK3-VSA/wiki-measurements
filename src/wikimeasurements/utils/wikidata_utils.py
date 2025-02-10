import os
import sys
import json
import time
import logging
import requests
import urllib.parse
from collections import defaultdict
import spacy
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
from wikimeasurements.utils.type_aliases import Facts, Labels, UnitConvData


nlp = spacy.load("en_core_web_md")
STOPWORDS = nlp.Defaults.stop_words
ENDPOINT_URL = "https://query.wikidata.org/sparql"
logger = logging.getLogger("logs/wikidata.log")

def write_to_json(content, filepath):

    with open(filepath, "w+", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)


def query_convert_and_repair(sparql):
    """Query and return the results converted to JSON. If the results are
    uncomplete due to a timeout, an attempt to recover valid JSON from
    the uncomplete results is started."""

    try:
        start = time.time()
        response_str = sparql.query().response.read().decode("utf-8")
        end = time.time()
        try:
            # Analogous to _convertJSON() in SPARQLWrapper
            results = json.loads(response_str)
            if not isinstance(results, dict):
                raise TypeError(type(results))

        except Exception as e:
            if "TimeoutException" in response_str:
                # Run into the 60 seconds timeout, hence the returned
                # JSON is uncomplete and we got a JSONDecodeError.
                logger.warning("⚠️\tRun into timeout.")
                # We attempt to cut the response at the last
                # complete record and close all the brackets.
                # First, remove the query string from the end,
                # since it also includes opening brackets.
                cut_here = response_str.rfind("SPARQL-QUERY")
                cut_here = response_str[:cut_here].rfind(", {")
                repaired_response = response_str[:cut_here] + "\n]\n}\n}"
                recovered_results = json.loads(repaired_response)
                if isinstance(recovered_results, dict):
                    logger.info(f"🛠️\tManaged to repair the uncomplete JSON response.")
                    results = recovered_results
                else:
                    raise TypeError(type(recovered_results))
            else:
                raise e

    except Exception as e:
        logger.error(e)
        results = {"results": {"bindings": []}}

    num_records = len(results["results"]["bindings"])
    logger.info(
        f"⏳️\tQuery took {end - start} seconds and yielded {num_records} records."
    )
    return results


def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (
        sys.version_info[0],
        sys.version_info[1],
    )
    # TODO: adjust user agent; see https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)    
    sparql.setReturnFormat(JSON)
    results = query_convert_and_repair(sparql)

    return results


def get_labels(
    uri_list,
    discard_abbr=False,
    threshold=250,
    pbar_descr="Processing URIs...",
):
    """Get all labels including alternative labels (that is, aliases)
    for a list of Wikidata items. Since the query length is limited,
    the full list is queried one chunk after another. The character
    limit seems to be somewhere around 6000 characters.

    :param uri_list: List of Wikidata items (e.g., ['wd:Q573', 'wd:Q577', ...])
    :type uri_list: list
    :param threshold: Max. number of items per query, defaults to 500
    :type threshold: int, optional
    :return: Dictionary of items and their corresponding labels including aliases (e.g.,
             {'http://www.wikidata.org/entity/Q628598':['Citroen DS3 WRC', 'DS3 WRC', 'Citroën DS3 WRC'], ...})
    :rtype: dict
    """

    if discard_abbr:
        abbreviation_filter = """
            OPTIONAL { ?item wdt:P246 ?chem_symbol . }
            OPTIONAL { ?item wdt:P1813 ?short_name . }
            FILTER (!bound(?chem_symbol) || !sameTerm(str(?itemLabel), str(?chem_symbol)))
            FILTER (!bound(?short_name) || !sameTerm(str(?itemLabel), str(?short_name)))
            """
    else:
        abbreviation_filter = ""

    LABEL_QUERY = (
        lambda items: f"""
        SELECT ?item ?itemLabel
        WHERE {{          
            VALUES ?item {{{items}}}
            ?item rdfs:label|skos:altLabel ?itemLabel .
            FILTER(LANG(?itemLabel) = "en") .
            {abbreviation_filter}
        }}
        """
    )

    aggregated_results = []
    uri_list = list(uri_list)

    # Split list of URIs into chunks in order to circumvate the "414 Request-URI Too Large" error
    chunks = [uri_list[x : x + threshold] for x in range(0, len(uri_list), threshold)]

    # Query labels and aliases for each chunk
    with tqdm(chunks) as pbar:
        pbar.set_description(pbar_descr)
        for chunk in pbar:
            uris = " ".join(chunk)
            results = get_results(ENDPOINT_URL, LABEL_QUERY(uris))
            aggregated_results += results["results"]["bindings"]

    # Transform results into output data structure
    labels = defaultdict(list)
    for result in aggregated_results:
        label = result["itemLabel"]["value"]
        labels[result["item"]["value"]].append(label)

    return labels


def get_all_quantitative_properties(cache_dir="./cached_data/", force=False):
    # Get all quantitative properties in Wikidata

    PROPERTY_QUERY = """       
        SELECT ?property ?propertyLabel
        WHERE {  
        ?property wikibase:propertyType wikibase:Quantity .
        ?property rdfs:label|skos:altLabel ?propertyLabel .
        FILTER(LANG(?propertyLabel) = "en") .
        }"""

    property_cache = os.path.join(cache_dir, "property_cache.json")

    if (
        not force
        and os.path.isfile(property_cache)
        and os.path.getsize(property_cache) > 0
    ):
        logger.info("Use properties written to cache before.")
        with open(property_cache, encoding="utf-8") as f:
            properties = json.load(f)
    else:
        # Create properties cache path
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)        

        logger.info("Collect all quantity properties of Wikidata...")
        property_results = get_results(ENDPOINT_URL, PROPERTY_QUERY)
        properties = defaultdict(list)
        for result in property_results["results"]["bindings"]:
            uri = result["property"]["value"]
            label = result["propertyLabel"]["value"]
            properties[uri].append(label)
        logger.info(f"Collected {len(properties)} quantity properties.\n")

    write_to_json(properties, property_cache)

    return properties


def group_concat_quantitative_statement(quantifications_results, property):
    nbr_statements_before = len(quantifications_results["results"]["bindings"])
    indexed_statements = defaultdict(list)
    for statement in quantifications_results["results"]["bindings"]:
        group_according_to = [
            statement.get("entity"),
            statement.get("value"),
            statement.get("unit"),
            statement.get("article"),
            statement.get("lowerbound"),
            statement.get("upperbound"),
        ]
        index = hash(str(group_according_to))
        indexed_statements[index].append(statement)

    for statement_group in indexed_statements.values():
        # Group according to:
        entity = statement_group[0].get("entity")
        value = statement_group[0].get("value")
        unit = statement_group[0].get("unit")
        article = statement_group[0].get("article")
        lowerbound = statement_group[0].get("lowerbound")
        upperbound = statement_group[0].get("upperbound")

        # Concatenate:
        qualifiers = []
        qualifier_values = []
        qualifier_units = []
        qualifier_lowerbounds = []
        qualifier_upperbounds = []
        qualifier_time_precisions = []
        for s in statement_group:
            quantifications_results["results"]["bindings"].remove(s)
            qualifiers.append(s.get("qualifier", {}).get("value"))
            qualifier_values.append(s.get("qualifier_value", {}).get("value"))
            qualifier_units.append(s.get("qualifier_unit", {}).get("value"))
            qualifier_lowerbounds.append(s.get("qualifier_lowerbound", {}).get("value"))
            qualifier_upperbounds.append(s.get("qualifier_upperbound", {}).get("value"))
            qualifier_time_precisions.append(
                s.get("qualifier_time_precision", {}).get("value")
            )

        grouped_results = {
            "article": article,
            "entity": entity,
            "property": {"type": "uri", "value": property},
            "value": value,
            "unit": unit,
            "qualifiers": [""] if qualifiers == [None] else qualifiers,
            "qualifier_values": [""]
            if qualifier_values == [None]
            else qualifier_values,
            "qualifier_units": [""] if qualifier_units == [None] else qualifier_units,
            "qualifier_lowerbounds": [""]
            if qualifier_lowerbounds == [None]
            else qualifier_lowerbounds,
            "qualifier_upperbounds": [""]
            if qualifier_upperbounds == [None]
            else qualifier_upperbounds,
            "qualifier_time_precisions": [""]
            if qualifier_time_precisions == [None]
            else qualifier_time_precisions,
        }

        if (lowerbound != None) and (upperbound != None):
            grouped_results.update({"lowerbound": lowerbound})
            grouped_results.update({"upperbound": upperbound})

        quantifications_results["results"]["bindings"].append(grouped_results)

    nbr_statements_after = len(quantifications_results["results"]["bindings"])
    print(
        f"Reduced number of statements by {nbr_statements_before-nbr_statements_after} by grouping and concatenating"
    )

    return quantifications_results


def get_quantitative_statements(
    properties, wiki_lang="en", cache_dir="./cached_data/", force=False, use_group_concat=False
):
    """Query quantitative statements from Wikidata"""

    if use_group_concat:
        # Also more elegant, using group_concat in the query results in a timeout
        QUANT_STATEMENT_QUERY = (
            lambda property_id, limit_threshold: f"""
            SELECT DISTINCT ?article ?entity ?value ?unit ?lowerbound ?upperbound         
                (group_concat(?qualifier;separator="🛁") as ?qualifiers) 
                (group_concat(?qualifier_value;separator="🛁") as ?qualifier_values) 
                (group_concat(?qualifier_unit;separator="🛁") as ?qualifier_units) 
                (group_concat(?qualifier_time_precision;separator="🛁") as ?qualifier_time_precisions)
                (group_concat(?qualifier_lowerbound;separator="🛁") as ?qualifier_lowerbounds) 
                (group_concat(?qualifier_upperbound;separator="🛁") as ?qualifier_upperbounds)             
            WHERE {{  
                # Fix property    
                VALUES (?property) {{(wd:{property_id})}} 
                
                # Get associated wikipedia article   
                ?article schema:about ?entity .        
                ?article schema:isPartOf <https://{wiki_lang}.wikipedia.org/> .
            
                # Consider statements which have a quantity as the object    
                ?entity    ?p  ?statement .
                ?statement ?ps ?valuenode .
                ?property  wikibase:claim          ?p ;
                        wikibase:statementValue ?ps .
                ?valuenode wikibase:quantityAmount ?value ;
                        wikibase:quantityUnit   ?unit .    
            
                # Get upper and lower bound, if applicable
                OPTIONAL {{     
                ?valuenode wikibase:quantityLowerBound ?lowerbound ;
                            wikibase:quantityUpperBound ?upperbound .
                }}
                    
                # Get qualifiers, if applicable
                OPTIONAL {{    
                    # Just some qualifier
                    ?statement ?qualifier ?qualifier_value .    
                    ?wdpq wikibase:qualifier ?qualifier . 

                    OPTIONAL {{
                        # A quantitative qualifier
                        ?statement ?pqv ?pqv_ .
                        ?wdpq wikibase:qualifierValue ?pqv .
                        OPTIONAL {{             
                            ?pqv_ wikibase:quantityUnit ?qualifier_unit .
                        }}
                        OPTIONAL {{
                            ?pqv_ wikibase:quantityLowerBound ?qualifier_lowerbound ;
                                wikibase:quantityUpperBound ?qualifier_upperbound .     
                        }}                    
                        OPTIONAL {{                          
                            ?pqv_ wikibase:timePrecision ?qualifier_time_precision .
                        }}
                        # (No precision for coordinate values, since it is mostly refered to locations in text by name)
                    }}    
                }}      
            }} 
            GROUP BY ?article ?entity ?value ?unit ?lowerbound ?upperbound 
            LIMIT {limit_threshold}
            """
        )
    else:
        QUANT_STATEMENT_QUERY = (
            lambda property_id, limit_threshold: f"""
            SELECT DISTINCT ?article ?entity ?value ?unit ?lowerbound ?upperbound ?qualifier ?qualifier_value ?qualifier_unit ?qualifier_lowerbound ?qualifier_upperbound ?qualifier_time_precision
            WHERE {{  
                # Fix property    
                VALUES (?property) {{(wd:{property_id})}} 
                
                # Get associated wikipedia article   
                ?article schema:about ?entity .        
                ?article schema:isPartOf <https://{wiki_lang}.wikipedia.org/> .
            
                # Consider statements which have a quantity as the object    
                ?entity    ?p  ?statement .
                ?statement ?ps ?valuenode .
                ?property  wikibase:claim          ?p ;
                        wikibase:statementValue ?ps .
                ?valuenode wikibase:quantityAmount ?value ;
                        wikibase:quantityUnit   ?unit .    
            
                # Get upper and lower bound, if applicable
                OPTIONAL {{     
                ?valuenode wikibase:quantityLowerBound ?lowerbound ;
                            wikibase:quantityUpperBound ?upperbound .
                }}
                    
                # Get qualifiers, if applicable
                OPTIONAL {{    
                    # Just some qualifier
                    ?statement ?qualifier ?qualifier_value .    
                    ?wdpq wikibase:qualifier ?qualifier . 

                    OPTIONAL {{
                        # A quantitative qualifier
                        ?statement ?pqv ?pqv_ .
                        ?wdpq wikibase:qualifierValue ?pqv .
                        OPTIONAL {{             
                            ?pqv_ wikibase:quantityUnit ?qualifier_unit .
                        }}
                        OPTIONAL {{
                            ?pqv_ wikibase:quantityLowerBound ?qualifier_lowerbound ;
                                wikibase:quantityUpperBound ?qualifier_upperbound .     
                        }}                    
                        OPTIONAL {{                          
                            ?pqv_ wikibase:timePrecision ?qualifier_time_precision .
                        }}
                        # (No precision for coordinate values, since it is mostly refered to locations in text by name)
                    }}    
                }}      
            }}        
            LIMIT {limit_threshold}
            """
        )

    facts_cache = os.path.join(cache_dir, "quantitative_statements_cache.json")

    if not force and os.path.isfile(facts_cache) and os.path.getsize(facts_cache) > 0:
        logger.info("Use quantitative statements written to cache before.")
        with open(facts_cache, encoding="utf-8") as f:
            return json.load(f)

    # Query one property at a time to lower the risk of running into a timeout
    results = []
    limited_properties = []

    # Threshold based on trials, where
    # wd:P1082 fails with 500_000, but is fine with 250_000,
    # wd:P6258 fails with 100_000 and 10_000.
    limit_threshold = 250_000

    logger.info(
        "Query Wikidata for quantitative statements one property at a time.\n☕️\tThis will take some time..."
    )
    last_saved_at = time.time()
    with tqdm(properties.items()) as pbar:
        for property, property_label in pbar:

            # Print status update
            property_id = property.lstrip("http://www.wikidata.org/entity/")
            property_label_sring = (
                property_id + " (" + " a.k.a. ".join(property_label) + ")"
            )
            pbar.set_description("Processing %s" % property_label_sring)

            # Query Wikidata
            query = QUANT_STATEMENT_QUERY(property_id, limit_threshold)
            quantifications_results = get_results(ENDPOINT_URL, query)

            # Print status update
            results_count = len(quantifications_results["results"]["bindings"])
            pbar.set_description("Retrieved %s results" % results_count)

            # Check if query ran into limit
            if results_count == limit_threshold:
                limited_properties.append(property_id)

            # Add result to aggregated results
            if use_group_concat:
                for row in quantifications_results["results"]["bindings"]:

                    # Add property to results dict
                    row["property"] = {"type": "uri", "value": property}

                    # Split concatenated qualifier information
                    row["qualifiers"] = row["qualifiers"]["value"].split("🛁")
                    row["qualifier_values"] = row["qualifier_values"]["value"].split(
                        "🛁"
                    )
                    row["qualifier_units"] = row["qualifier_units"]["value"].split("🛁")
                    row["qualifier_lowerbounds"] = row["qualifier_lowerbounds"][
                        "value"
                    ].split("🛁")
                    row["qualifier_upperbounds"] = row["qualifier_upperbounds"][
                        "value"
                    ].split("🛁")
                    row["qualifier_time_precisions"] = row["qualifier_time_precisions"][
                        "value"
                    ].split("🛁")

            else:
                quantifications_results = group_concat_quantitative_statement(
                    quantifications_results, property
                )

            results += quantifications_results["results"]["bindings"]

            if time.time() - last_saved_at > 30 * 60:  # 30 minutes
                write_to_json(results, facts_cache)
                last_saved_at = time.time()

    # Print status update
    if len(limited_properties) > 0:
        logger.warning(
            "❌️\tAll properties which yield limited results: "
            + ", ".join(limited_properties)
            + "\n"
        )
    else:
        logger.info("✔️\tNo propterty yielded limited results!\n")

    write_to_json(results, facts_cache)

    return results


def get_page_mappings(page_urls, wiki_lang="en", cache_dir="./cached_data/", use_cache=True):
    """Create mapping between page urls, page titles and page ids"""

    mapping_cache = os.path.join(cache_dir, "page_mappings_cache.json")

    if (
        use_cache
        and os.path.isfile(mapping_cache)
        and os.path.getsize(mapping_cache) > 0
    ):
        with open(mapping_cache, encoding="utf-8") as f:
            cached_mappings = json.load(f)

        id_to_url = cached_mappings["id_to_url"]
        url_to_id = cached_mappings["url_to_id"]
        id_to_title = cached_mappings["id_to_title"]
        new_page_urls = [
            url
            for url in page_urls
            if urllib.parse.unquote(url)  # unescape %xx special chars
            not in url_to_id.keys()
        ]

        # TODO: remove URLs for mapping attempts that already failed once from new_page_urls
        # failed_mappings = cached_mappings["failed_mappings"]

    else:
        id_to_url = {}
        url_to_id = {}
        id_to_title = {}
        new_page_urls = page_urls

    # Split list of URLs into chunks in order to circumvate the request limit
    api_limit = 50
    chunks = [
        new_page_urls[x : x + api_limit]
        for x in range(0, len(new_page_urls), api_limit)
    ]

    wiki_url = f"https://{wiki_lang}.wikipedia.org/wiki/"
    with tqdm(chunks) as pbar:
        pbar.set_description("Get page IDs from page titles")
        failed_mappings = {}
        failed_mappings["no_page_id_for_given_title"] = []
        failed_mappings["wrong_namespace"] = []
        for chunk in pbar:
            page_titles = [url.removeprefix(wiki_url) for url in chunk]
            page_titles_string = "|".join(page_titles)
            # Note that redirects are resolved
            api_url = f"https://{wiki_lang}.wikipedia.org/w/api.php?action=query&titles={page_titles_string}&format=json&redirects"
            results = requests.get(api_url).json()

            title_normalized = {}
            normalizations = results["query"].get("normalized")
            if normalizations is not None:
                [
                    title_normalized.update({row["to"]: row["from"]})
                    for row in normalizations
                ]

            title_redirected = {}
            redirects = results["query"].get("redirects")
            if redirects is not None:
                [title_redirected.update({row["to"]: row["from"]}) for row in redirects]

            for result in results["query"]["pages"].values():
                if "missing" in result.keys():
                    # Example: {'ns': 0, 'title': 'Yotpo', 'missing': ''}
                    failed_mappings["no_page_id_for_given_title"].append(
                        result["title"]
                    )

                    continue
                elif result["ns"] != 0:
                    failed_mappings["wrong_namespace"].append(result["title"])
                    logger.warning(
                        f'The namespace for {result["title"]} is {result["ns"]} and not 0 (main/article)'
                    )
                    continue

                id = result["pageid"]
                title = result["title"]
                redirect_title = title_redirected.get(title)
                normalized_title = title_normalized.get(
                    title if redirect_title is None else redirect_title
                )
                url_title = title if normalized_title is None else normalized_title
                url = wiki_url + url_title
                id_to_url.update({id: url})
                url_to_id.update({url: id})
                id_to_title.update({id: title})

    if len(failed_mappings["no_page_id_for_given_title"]) > 0:
        missing_id_list_str = "\n\t* " + "\n\t* ".join(
            set(failed_mappings["no_page_id_for_given_title"])
        )
        logger.warning(
            "Missing page IDs for the pages with the following titles. "
            + "Most likely, there is no article in the English Wikipedia with the given title if a page ID is missing."
            + missing_id_list_str
        )

    if len(failed_mappings["wrong_namespace"]) > 0:
        wrong_namespace_list_str = "\n\t* " + "\n\t* ".join(
            set(failed_mappings["wrong_namespace"])
        )
        logger.warning(
            "The namespaces of the pages with the follwing titles do not equal 0 (main/article):"
            + wrong_namespace_list_str
        )

    page_mappings = {
        "id_to_url": id_to_url,
        "url_to_id": url_to_id,
        "id_to_title": id_to_title,
        "failed_mappings": failed_mappings,
    }

    with open(mapping_cache, "w", encoding="utf-8") as f:
        json.dump(page_mappings, f, ensure_ascii=False, indent=4)

    return page_mappings


def shorten_using_prefix(uri_list):
    """Shorten Wikidata URIs in a list of URIs by using prefixes
    instead of the full URI.

    :param uri_list: List of URIs in the Wikidata domain
    :type uri_list: list
    :return: List of URIs using prefixes
    :rtype: list
    """
    prefix_uri_list = []
    for uri in uri_list:
        if uri == None or uri[11:].startswith("wikipedia.org/"):
            continue
        elif uri.startswith("http://www.wikidata.org/entity/"):
            prefix_uri = "wd:" + uri.lstrip("http://www.wikidata.org/entity/")
        elif uri.startswith("http://www.wikidata.org/prop/qualifier/"):
            prefix_uri = "pq:" + uri.lstrip("http://www.wikidata.org/prop/qualifier/")
        else:
            logger.warning("Unknown PREFIX for given URI: " + uri)
            continue

        prefix_uri_list.append(prefix_uri)

    return prefix_uri_list


def convert_wikidata_units(unit_in: str, wikidata_unit_conversions: dict[dict[str]]):
    """Get data for unit conversion, where

    new_value [to_unit] = old_value [unit_in] * conv_factor_numerator / conv_factor_denominator
    (e.g., 2200 mm = 2.2 m * 1/0.001)

    The conversion factors of all conversions defined for the given unit are returned,
    including the conversion to itself.

    wikidata_unit_conversions is a dict from "unitConversionConfig.json".
    "unitConversionConfig.json" is the configuration file for unit conversion in Mediawiki.
    It can be obtained from:
    https://gerrit.wikimedia.org/r/plugins/gitiles/operations/mediawiki-config/+/master/wmf-config/unitConversionConfig.json

    """
    unit_in_dict = wikidata_unit_conversions.get(unit_in)
    conv_quantity = []
    if unit_in_dict is not None:
        for q_id, unit_out_dict in wikidata_unit_conversions.items():
            if unit_out_dict["siLabel"] == unit_in_dict["siLabel"]:
                quantity_out = {
                    "conv_factor_numerator": unit_in_dict["factor"],
                    "conv_factor_denominator": unit_out_dict["factor"],
                    "to_unit": q_id,
                }
                conv_quantity.append(quantity_out)

    return conv_quantity


def get_unit_conversion_data(units_in: list[str]):

    with open(
        "src/wikimeasurements/static_resources/unitConversionConfig.json",
        encoding="utf-8",
    ) as f:
        wikidata_unit_conversions = json.load(f)

    unit_conversion_data = {}
    for unit_in in units_in:
        unit_in_q_id = unit_in.removeprefix("wd:")
        units_out = convert_wikidata_units(unit_in_q_id, wikidata_unit_conversions)
        if len(units_out) == 0:
            # Add conversion to itself, for units not in unitConversionConfig.json
            units_out = [
                {
                    "conv_factor_numerator": "1",
                    "conv_factor_denominator": "1",
                    "to_unit": unit_in_q_id,
                }
            ]
        unit_conversion_data.update({unit_in: units_out})

    return unit_conversion_data


def load_wikidata_knowledge(path: str) -> tuple[Facts, int, list[int], Labels, UnitConvData]:
    with open(path, encoding="utf-8") as f:
        wikidata_knowledge = json.load(f)

    # global LABELS
    LABELS = wikidata_knowledge["labels"]
    UNIT_CONVERSION_DATA = wikidata_knowledge["unit_conversion_data"]
    facts = wikidata_knowledge["quantitative_statements"]
    pages_with_facts = [int(id) for id in set(facts.keys())]
    pages_with_facts.sort()

    nbr_facts = sum(len(f) for f in facts)
    nbr_pages = sum(1 for fact in facts)
    print(
        f"Loaded {nbr_facts} Wikidata facts associated to {nbr_pages} different Wikipedia pages."
    )

    return facts, nbr_facts, pages_with_facts, LABELS, UNIT_CONVERSION_DATA


def remove_stopwords_and_duplicates(labels: dict):
    for uri, aliases in labels.items():
        filtered_aliases = []
        distinct_aliases = set(aliases)
        for alias in distinct_aliases:
            if alias.lower() not in STOPWORDS:
                filtered_aliases.append(alias)
            else:
                print("Exclude labels based on stopwords: " + alias)

        labels[uri] = filtered_aliases

    return labels
