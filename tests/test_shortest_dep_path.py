import pytest
import spacy
from wikimeasurements.utils.nlp_utils import shortest_path


def test_shortest_path():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The great eagle has a wingspan of approx 5 m, which is 90 cm more than that of the wood eagle.")

    sent = doc[0:]
    quantity_annotation_1 = (41, 44)
    quantity_annotation_2 = (55, 60)
    entity_annotations = [(4, 15), (83, 93)]
    same_hops = [(4, 15), (71, 75)]

    assert shortest_path(sent, entity_annotations, quantity_annotation_1) == [0]
    assert shortest_path(sent, entity_annotations, quantity_annotation_2) == [1]
    assert shortest_path(sent, same_hops, quantity_annotation_1) == [0, 1]

    doc = nlp("The great eagle has a wingspan of approx 5 m, 90 cm more than the wood eagle.")
    sent = doc[0:]
    quantity_annotation_1 = (41, 44)
    quantity_annotation_2 = (46, 51)
    entity_annotations = [(4, 15), (66, 76)]

    assert shortest_path(sent, entity_annotations, quantity_annotation_1) == [0]
    assert shortest_path(sent, entity_annotations, quantity_annotation_2) == [0, 1]


test_shortest_path()