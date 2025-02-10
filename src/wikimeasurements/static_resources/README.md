# Description of the files in this directory

## `basic_number_words.json`
Various number words including cardinals (fourth or quartary), ordinals (seventh), large magnitudes (e.g., octillion), and denominators (e.g., quarter).        

## `colloquial_number_words.json`
Various quantity terms compiled from tables within Hanauer et al. „Complexities, Variations, and Errors of Numbering within Clinical Notes: The Potential Impact on Information Extraction and Cohort-Identification“, 2019. (https://doi.org/10.1186/s12911-019-0784-1). The data is licensed under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/).

## `physical_constants.json`
A list of names of physical constants which was compiled by quering [Wikidata](https://query.wikidata.org/) with the following query and manually filtering the results. As the data is based on Wikidata, it is licensed under [CC0](https://creativecommons.org/publicdomain/zero/1.0/).
```SQL
SELECT ?item ?itemLabel ?altLabel
WHERE {
    ?item wdt:P31 wd:Q173227 .
    OPTIONAL { ?item skos:altLabel ?altLabel . FILTER (lang(?altLabel) = \"en\") }
    SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". }
}
```

## `unitConversionConfig.json`
unitConversionConfig.json is the configuration file for unit conversion in Mediawiki. It can be obtained from https://github.com/wikimedia/operations-mediawiki-config/blob/master/wmf-config/unitConversionConfig.json. Please respect their License.
