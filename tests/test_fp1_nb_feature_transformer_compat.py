"""Unpickling FPFeatureTransformer objects saved before include_negative_context existed.

The flag arrived in 14f1e69 with default True. A pickle saved before it (registry FP
v2.4.0, the served model) has no such attribute, and transform() reads it (D028).
"""

import pickle

import numpy as np
import pytest

from src.fp1_nb.feature_transformer import FPFeatureTransformer

TEXTS = [
    "Nike announced a new recycled running shoe line for the spring season.",
    "The puma was spotted near the ranch, wildlife officials said on Monday.",
    "Adidas signed a sponsorship deal with the national football team.",
    "Patagonia region tourism grew as visitors hiked the southern glaciers.",
    "Under Armour cut its sales forecast after weak apparel demand in China.",
    "Columbia University researchers published a study on river pollution.",
]


def _fitted(include_negative_context: bool) -> FPFeatureTransformer:
    transformer = FPFeatureTransformer(
        method="tfidf_lsa_proximity",
        min_df=1,
        max_df=1.0,
        lsa_n_components=2,
        include_metadata_features=False,
        include_negative_context=include_negative_context,
    )
    return transformer.fit(TEXTS)


def _legacy_round_trip(transformer: FPFeatureTransformer) -> FPFeatureTransformer:
    """Pickle a transformer whose state lacks include_negative_context, as pre-14f1e69 pickles do.

    Only objects this test built are pickled and loaded.
    """
    legacy = pickle.loads(pickle.dumps(transformer))
    del legacy.__dict__["include_negative_context"]
    payload = pickle.dumps(legacy)
    assert b"include_negative_context" not in payload
    return pickle.loads(payload)


@pytest.fixture(scope="module")
def with_negative_context() -> FPFeatureTransformer:
    return _fitted(include_negative_context=True)


@pytest.fixture(scope="module")
def without_negative_context() -> FPFeatureTransformer:
    return _fitted(include_negative_context=False)


def test_legacy_pickle_restores_negative_context_as_true(with_negative_context):
    loaded = _legacy_round_trip(with_negative_context)

    assert loaded.include_negative_context is True
    assert loaded.get_params()["include_negative_context"] is True


def test_legacy_pickle_transforms_with_the_negative_context_columns(
    with_negative_context, without_negative_context
):
    loaded = _legacy_round_trip(with_negative_context)

    features = loaded.transform(TEXTS)
    expected = with_negative_context.transform(TEXTS)
    # The flag gates extra columns, so the width alone distinguishes on from off.
    assert features.shape[1] > without_negative_context.transform(TEXTS).shape[1]
    np.testing.assert_array_equal(features, expected)


def test_pickle_carrying_false_stays_false(without_negative_context):
    loaded = pickle.loads(pickle.dumps(without_negative_context))

    assert loaded.include_negative_context is False
    np.testing.assert_array_equal(
        loaded.transform(TEXTS), without_negative_context.transform(TEXTS)
    )


def test_legacy_default_does_not_follow_the_init_default(monkeypatch, with_negative_context):
    """A later change to the __init__ default must not change what a legacy pickle means."""
    original_init = FPFeatureTransformer.__init__
    defaults = list(original_init.__defaults__)
    names = original_init.__code__.co_varnames[1 : original_init.__code__.co_argcount]
    defaults[names.index("include_negative_context") - (len(names) - len(defaults))] = False
    monkeypatch.setattr(original_init, "__defaults__", tuple(defaults))
    assert FPFeatureTransformer().include_negative_context is False

    loaded = _legacy_round_trip(with_negative_context)

    assert loaded.include_negative_context is True
