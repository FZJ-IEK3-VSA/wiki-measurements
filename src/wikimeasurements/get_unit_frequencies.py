import json
from collections import Counter
from argparse import ArgumentParser
from tqdm import tqdm
from wikimeasurements.utils.wikidata_utils import load_wikidata_knowledge
from wikimeasurements.create_datasets import get_fulltext
import spacy
from spacy.matcher import PhraseMatcher
from wikimeasurements.utils.wikidata_utils import write_to_json


nlp = spacy.load("en_core_web_md")
nlp.add_pipe("lower_lemmas")
STOPWORDS = nlp.Defaults.stop_words

def count_unit_occurances(
    unit_matcher: PhraseMatcher, distinct_units: list[str], max_pages: int
) -> list[tuple[str, int]]:

    # Initialize ouput dict
    unit_counts = dict.fromkeys(distinct_units, 0)

    # Iterate over Wikipedia articles and count the matched units
    with open(args.input_path) as f:
        pbar = tqdm(total=max_pages)
        for line in f:

            # Get wiki text
            wiki_page = json.loads(line)
            full_text, quantities, _ = get_fulltext(wiki_page)

            with nlp.select_pipes(disable=["parser", "ner"]):
                if mode == "fulltext":
                    docs = [nlp(full_text)]

                elif mode == "quantities":
                    if len(quantities) == 0:
                        continue
                    else:
                        docs = list(nlp.pipe(quantities))

            # Match units and increase counts
            for doc in docs:
                matches = unit_matcher(doc)
                for match_id, _, _ in matches:
                    tag = nlp.vocab.strings[match_id]
                    unit_counts[tag] += 1

            pbar.update(1)

            # Stop after a certain number of pages
            if pbar.n == max_pages:
                break

    # Print some results and save to file
    c = Counter(unit_counts)
    freqs = c.most_common()

    return freqs


def print_stats(freqs: list[tuple[str, int]]):

    print("\nThe 💯 most frequent units:")
    for (Q, count) in freqs[0:100]:
        unit = "http://www.wikidata.org/entity/" + Q
        labels = LABELS["units"].get(unit)
        labels_str = ", ".join(labels)
        print("* " + str(count) + ":\t" + labels_str)

    print("\nThe 💯 less frequent units:")
    for (Q, count) in freqs[-100:]:
        unit = "http://www.wikidata.org/entity/" + Q
        labels = LABELS["units"].get(unit)
        labels_str = ", ".join(labels)
        print("* " + str(count) + ":\t" + unit)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="./workflow/intermediates/parsed_wikipedia_dump.json",
        help="""Path to the Wikipedia dump parsed with parse_wikipedia_dump.py""",
    )
    parser.add_argument(
        "--wikidata_facts_path",
        default="./workflow/intermediates/wikidata_quantitative_statements.json",
        help="""Path to quantitative statements queried from Wikidata with query_wikidata.py""",
    )
    parser.add_argument(
        "--outfile",
        default="./workflow/intermediates/unit_freqs.json",
        help="""Path of output file (e.g., './unit_freqs.json')""",
    )
    parser.add_argument(
        "--nbr_pages_to_analyze",
        default=[10_000, 50_000],
        nargs="+",
        type=int,
        help="""The total number of Wikipedia pages to analyze for units. 
        The first number applies to the 'fulltext' mode, the second number 
        applies to the 'quantities' mode.""",
    )
    parser.add_argument(
        "--omit_fulltext",
        action="store_true",
        help="""Whether frequencies should be calculated based on 
        occurances within Wikipedia articles or not.""",
    )
    parser.add_argument(
        "--omit_quantities",
        action="store_true",
        help="""Whether frequencies should be calculated based on
        the content of the convert templates or not.""",
    )
    args = parser.parse_args()

    # Get distant supervision knowledge for context dataset
    _, _, _, LABELS, UNIT_CONVERSION_DATA = load_wikidata_knowledge(
        args.wikidata_facts_path
    )

    # Get all distinct units
    all_units = []
    for unit_in in UNIT_CONVERSION_DATA.values():
        for conversion_dict in unit_in:
            all_units.append(conversion_dict["to_unit"])

    distinct_units = list(set(all_units))

    # Prepare unit matcher
    no_english_labels = []
    unit_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA", validate=False)
    for Q in distinct_units:
        unit = "http://www.wikidata.org/entity/" + Q
        unit_aliases = LABELS["units"].get(unit)
        if unit_aliases is None:
            no_english_labels.append(Q)
        else:
            # Ensure units are not in stopword list
            # (although this is already done within the wikidata query script)
            unit_aliases = [l for l in unit_aliases if l.lower() not in STOPWORDS]
            with nlp.select_pipes(disable=["parser", "ner"]):
                unit_phrases = list(nlp.pipe(unit_aliases))
                unit_matcher.add(Q, unit_phrases)

    # Remove units wihout labels for counting
    print(str(len(no_english_labels)) + " units do not have an English label.")
    for u in no_english_labels:
        distinct_units.remove(u)

    # Get total number of pages for process bar
    total_page_count = 0
    with open(args.input_path) as f:
        for p in f:
            total_page_count += 1

    # Ensure that the number of pages to analyze is smaller
    # than the total avaible number of pages
    assert not any(total_page_count < pages for pages in args.nbr_pages_to_analyze)

    # The frequencies are either based on occurances within Wikipedia articles (fulltext)
    # or only within convert templates in Wikipedia articles (quantities).
    modes = []
    if not args.omit_fulltext:
        modes.append("fulltext")
    if not args.omit_quantities:
        modes.append("quantities")

    results = {}
    for mode, max_pages in zip(modes, args.nbr_pages_to_analyze):
        print("\nAnalyzing " + mode + "...")

        freqs = count_unit_occurances(unit_matcher, distinct_units, max_pages)
        print_stats(freqs)

        result = {
            "units_without_english_labels": no_english_labels,
            "frequencies_of_units_with_label": freqs,
        }
        results.update({mode: result})

    print("Write results to " + args.outfile)
    write_to_json(results, args.outfile)

    print("Finished! ✔️")
