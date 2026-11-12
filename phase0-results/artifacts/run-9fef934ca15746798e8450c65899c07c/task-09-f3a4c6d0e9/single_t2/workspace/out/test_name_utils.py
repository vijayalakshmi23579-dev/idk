import sys
sys.path.insert(0, '/tmp/phase0-case-a2wmnzw5/out')
from name_utils import normalize_name

def test_normalize_name_ada_lovelace():
    result = normalize_name('  Ada   LOVELACE ')
    assert result == 'ada lovelace', f"Expected 'ada lovelace', got '{result}'"

if __name__ == '__main__':
    test_normalize_name_ada_lovelace()
    print('Test passed!')