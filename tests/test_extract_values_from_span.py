import pytest
import spacy
from wikimeasurements.utils.nlp_utils import extract_values_from_span, inline_tags_to_char_offsets
from wikimeasurements.utils.quantity_utils import get_basic_number_words


def test_extract_values_from_span():
    """Test if extract_values_from_span correctly identifies numbers in text."""

    test_strings = [
        ["It covers an area of 🍏191,123🍏 sq mi, with a population of 🍏15.9 million🍏 as of 2021."],
        ["It is 🍏9.1🍏 km long and 🍏9.3🍏 km wide."],
        ["The Forschungszentrum Jülich is a member of the Helmholtz Association."],
        ["The building costs $🍏32,241,700🍏 to build including the demolition of the roof part."],
        ["The geographical mile is determined by 🍏1🍏 minute of arc along the Earth's equator."],        
        ["Gallium has the atomic number 🍏31🍏."],        
        ["The national census of 2020 recorded its population as approximately 🍏1,213,456🍏."],
        ["The total area is generally stated as being 🍏1,988,213🍏 sq mi."],
        ["Earth's mass is around 🍏5,900🍏 Yg."],
        ["The Earth is an ellipsoid and has a circumference of ~ 🍏40,000🍏 km."],
        ["The highest temperature ever recorded was 🍏39.2🍏 °C on April 01, 1992."],
        ["Until 1964, the bridge was the longest suspension bridge with 🍏4200🍏 feet and 🍏67🍏 m clearance above high water."],
        ["It is an alga which may grow up in length to 🍏12🍏 metres (🍏39🍏 ft)."],
        ["Numbers can be written in different ways, e.g., 🍏270🍏 can be written as 🍏2.7×102🍏 or 🍏27×101🍏 or 🍏270×100🍏."],
        ["🍏350🍏 can be written as 🍏3.5×10^2🍏 or 🍏35×10^1🍏 or 🍏350×10^0🍏."],
        ["🍏350🍏 can be written as 🍏3.5 10²🍏 or 🍏35 10^1🍏 or 🍏350 10^0🍏."],
        ["🍏350🍏 can be written as 🍏3.5*10^2🍏 or 🍏35*10^1🍏 or 🍏350*10^0🍏."],
        ["🍏3500🍏 can be written as 🍏35.0×10^2🍏 or 🍏350×10^1🍏 or 🍏3,500×10^0🍏."],
        ["🍏One🍏, 🍏two🍏 and 🍏three🍏"],
    ]

    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("custom_sentencizer", before="parser")
    nlp.add_pipe("lower_lemmas")

    with nlp.select_pipes(disable=["merge_entities", "ner"]):
        WIKI_ORDINALS_LOWER = get_basic_number_words(
            numeral_type="ordinals_minus_denominator_intersection"
        )
        WIKI_ORDINAL_LEMMAS = get_basic_number_words(numeral_type="ordinals")
        WIKI_ORDINAL_LEMMAS = list(nlp.pipe(WIKI_ORDINAL_LEMMAS))
        WIKI_ORDINAL_LEMMAS = list(set([tokens[0].lemma for tokens in WIKI_ORDINAL_LEMMAS]))

    for texts in test_strings:
        
        all_texts = []
        num_offsets_solution = []
        for sentence in texts:
            clean_text, num_offset = inline_tags_to_char_offsets(sentence)
            all_texts.append(clean_text)
            num_offsets_solution.append(num_offset["value"])
        
        full_text = " ".join(all_texts)
        
        with nlp.select_pipes(disable=["merge_entities"]):            
            doc = nlp(full_text)

        num_offsets = []
        for sent in doc.sents:
            num_offsets.append(extract_values_from_span(
                sent,
                doc,
                consider_ordinals=False,
                WIKI_ORDINAL_LEMMAS=WIKI_ORDINAL_LEMMAS,
                WIKI_ORDINALS_LOWER=WIKI_ORDINALS_LOWER,
            ))

        assert num_offsets == num_offsets_solution


if __name__ == "__main__":
    test_extract_values_from_span()
