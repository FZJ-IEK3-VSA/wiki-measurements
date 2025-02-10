# Run individual parts of the workflow
If you are interested in running the full workflow using Snakemake you are redirected to the [`README.md`](.) of the root directory. If you, however, want to run each step of the workflow separately without Snakemake you can follow the explanations below.


## Download an English Wikipedia dump
Download the Wikipedia dump of your choice. You can download English Wikipedia dumps [here](https://dumps.wikimedia.org/enwiki/), German Wikipedia dumps [here](https://dumps.wikimedia.org/dewiki/), and so on. That said, the workflow currently only supports English language.

As an example, we use the latest Englisch Wikipedia dump (approx. 18 GB compressed and 80 GB unzipped, but there is no need to unzip it).

```bash
wget -c 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2' -P 'workflow/input/enwiki-latest-pages-articles-multistream.xml.bz2'
```
⚠️ Don't try to open the whole file! It probably won't fit into memory. The parser will stream through it. If you want to take a look at the head of the file:
```bash
bzcat 'workflow/input/enwiki-latest-pages-articles-multistream.xml.bz2' | head -n 50
```

## Parse Wikipedia dump
If you checked the head of the Wikipedia dump you just downloaded, you may have noticed that the text consists of some gibberish. This is MediaWiki markup. In order to create the datasets we first have to parse the MediaWiki markup to clean text. 

In addition, the creation of the quantity dataset requires expanding the convert template calls, that is, for a convert template call such as `{{convert|60-62.5|m|ft+royal cubit|0|abbr=on}}` within the MediaWiki markup ([as seen in the edit mode for Wikipedia articles](https://en.wikipedia.org/w/index.php?title=Pyramid_of_Djoser&action=edit)), run the convert module and display the resulting string such as `60–62.5 m (197–205 ft; 115–119 cu)` ([as seen in the Wikipedia article](https://en.wikipedia.org/wiki/Pyramid_of_Djoser)) instead.

To run the parser which also takes care of convert template expansion, enter the following command:
```bash
python -m ./src/parse_wikipedia_dump --dump_path 'workflow/input/enwiki-latest-pages-articles-multistream.xml.bz2' --outfile 'workflow/intermediates/parsed_wikipedia_dump.json' --output_mode 'convert'
```

The parsed dump is an intermediate dataset which is later used to create the quantity and measurement context dataset. If you just want to parse a Wikipedia dump without expanding the convert template (which is required to run the following steps of the pipeline!), change the `--output_mode` to `raw`. 

## Query Wikidata for quantitative facts
The creation of the measurement context dataset is based on distant supervision, where matches with a knowledge base are treated as (somewhat noisy) groundtruth. Our knowledge base are quantitative facts from Wikidata. To query Wikidata for quantitative facts and collect them into an intermediate dataset run the following command:

```bash
python -m ./src/query_wikidata --outfile 'workflow/intermediates/wikidata_quantitative_statements.json'--cache_dir 'workflow/intermediates/wikidata_cache/' --output_mode 'production'
```
Since the process takes some time (we are accessing the public Wikidata SPARQL endpoint), parts of the results, such as labels of items, are cached. When re-running the query script, those labels etc. do not have to be queried again but are read from disk. You can ignore the cache and query the information nonetheless by appending `--force`. When changing `--output_mode` to `debug`, only quantitative statements of four properties are queried for a quick run.


## Get usage statistics of units in Wikipedia 
Creating the measurement context dataset additionally requires information on the occurance frequency of units of measurement in both the full Wikipedia articles and the convert template. To get this information, run the following command:
```bash
python -m ./src/get_unit_frequencies --input_path 'workflow/intermediates/parsed_wikipedia_dump.json' --wikidata_facts_path 'workflow/intermediates/wikidata_quantitative_statements.json' 
--outfile 'workflow/intermediates/unit_freqs.json' --nbr_pages_to_analyze 10_000 50_000'
```
When appending `--omit_fulltext` frequencies are not calculated based on the fulltext of the Wikipedia articles. When appending `--omit_quantities` frequencies are not calculated based on only the convert templates within the Wikipedia articles. The total number of Wikipedia pages to analyze is specified with `--nbr_pages_to_analyze`, where the first number applies to the 'fulltext' mode, the second number applies to the 'quantities' mode.


## Create datasets
To create the quantity span identification dataset and measurement context extraction dataset run the following command:
```bash
python -m ./src/create_datasets --input_path 'workflow/intermediates/parsed_wikipedia_dump.json' --outfile 'workflow/output/measurement_dataset.json' --wikidata_facts_path 'workflow/intermediates/wikidata_quantitative_statements.json' --unit_freqs_path 'workflow/intermediates/unit_freqs.json' --output_mode 'production'
```
By appanding `--omit_quantity_dataset_creation` and `--omit_context_dataset_creation` you skip the creation of the quantity dataset and measurement context dataset, respectively. Again, running it with `--output_mode` to `debug` provides additional information valuable for debugging.


# Ouput data formats
## Intermediate data
* parsed_wikipedia_dump.json
    
    You may have noticed that many Page ID's are missing in the ouput. Those pages are redirects and are filtered out. 

    * Using parse_wikipedia_dump.py with '--output_mode convert' (default):
        ```JSON
        {"id": 633,  "title": "Algae", "text": [["They may grow up to "], [" in length. Most are aquatic..."], [...], [...]], "quantities": [["50 metres", "160 ft", "50 metres (160 ft)"], [...], [...]]} 
        {"id": 633, ... }
        {"id": 639, ... }
        ```
    * Using parse_wikipedia_dump.py with '--output_mode raw' the page contents remain in raw MediaWiki markup. This output mode is not used in the above depicted pipeline for the creation of measurement extraction datasets but might be useful for other purposes. In raw mode the text consists of a lot of gibberish. This is MediaWiki's text markup. For cleaning the markup you may use [WikiExtractor](https://github.com/attardi/wikiextractor), [mwparserfromhell](https://github.com/earwig/mwparserfromhell) or [other parsers](https://www.mediawiki.org/wiki/Alternative_parsers).
        ```JSON
        {"id": 12, "title": "Anarchism", "text": "{{short description|Political philosophy and movement ..."}
        {"id": 25, "title": "Autism", "text": "{{Short description|Neurodevelopmental disorder involving social communication difficulties and repetitive behavior ..."}
        {"id": 39, "title": "Albedo", "text": "{{Short description|Ratio of reflected radiation to incident radiation ..."}
        ```

* wikidata_quantitative_statements.json


## Final datasets
* quantity_dataset.json
    ```JSON
    {"sentence": "Seaweeds grow mostly in shallow marine waters, under 100 m (330 ft) deep; however, ...", "quantities": [[53, 67], [148, 164]]}
    {"sentence": ...}
    ```
* measurement_context_dataset.json
  ```JSON
    {"id": 680, "title": "Aardvark", "context_dataset": [{"sentence": "Aardvarks pair only during the breeding season; after a gestation period of seven months, one cub weighing around 1.7–1.9 kilograms is born...", "annotations": {"entity": [[0, 9]], "property": [[56, 72]], "value": [[76, 81]], "unit": [[82, 88]]}, "wikidata_fact": {"entity": "http://www.wikidata.org/entity/Q46212", "property": "http://www.wikidata.org/entity/P3063", "value": "7", "value_lowerbound": "6", "value_upperbound": null, "unit": "http://www.wikidata.org/entity/Q5151", "qualifiers": [""], "qualifier_values": [""], "qualifier_lowerbounds": [""], "qualifier_upperbounds": [""], "qualifier_units": [""], "qualifier_time_precisions": [""], "article": "https://en.wikipedia.org/wiki/Aardvark"}, "context": {"before": ["## Reproduction.\n\n"], "after": ["When born, ..."]}, "weak_accept_reasons": []}]}
    ```

# Some hints
* You can deduce a page URL from the page ID by the following schema ([example](https://en.wikipedia.org/?curid=1091669)): `https://en.wikipedia.org/?curid={ID}` 
* You can ask the Wikipedia API for the page ID for a given page title ([example](https://en.wikipedia.org/w/api.php?action=query&titles=Forschungszentrum_Jülich&format=json)): `https://en.wikipedia.org/w/api.php?action=query&titles={title}&format=json` 


# Resources
* Wikipedia
    * [Wikipedia API help](https://en.wikipedia.org/w/api.php?action=help)
    * [Scribunto/Lua extension reference manual](https://www.mediawiki.org/wiki/Extension:Scribunto/Lua_reference_manual)
    * [Convert template documentation](https://en.wikipedia.org/wiki/Template:Convert)
    * [Convert template help](https://en.wikipedia.org/wiki/Help:Convert)
* Wikidata
    * [Data model](https://www.mediawiki.org/wiki/Wikibase/Indexing/RDF_Dump_Format#Data_model)
    * [Query limits](https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual#Query_limits)
    * [Query optimization](https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/query_optimization)
    * [Unit conversion, precision and coordinates](https://en.wikibooks.org/wiki/SPARQL/WIKIDATA_Precision,_Units_and_Coordinates)
* Parsers    
    * [WikiExtractor](https://github.com/attardi/wikiextractor)
    * [mwparserfromhell](https://github.com/earwig/mwparserfromhell)
    * [Alternative parsers](https://www.mediawiki.org/wiki/Alternative_parsers)