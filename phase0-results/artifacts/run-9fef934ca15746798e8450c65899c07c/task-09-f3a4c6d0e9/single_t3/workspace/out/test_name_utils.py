from name_utils import normalize_name


def test_normalize_name_ada_lovelace():
    assert normalize_name('  Ada   LOVELACE ') == 'ada lovelace'