"""Focused tests for name_utils.normalize_name."""

from name_utils import normalize_name


def test_normalize_name_trims_collapses_and_casefolds():
    assert normalize_name("  Ada   LOVELACE ") == "ada lovelace"
