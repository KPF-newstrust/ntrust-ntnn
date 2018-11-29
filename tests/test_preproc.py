from ntnn.preproc.nl import char_ids, sent_ids
from ntnn.preproc.nl import CHO_ID, JUNG_ID, JONG_ID


def test_char_ids():
    assert char_ids('각') == [CHO_ID, JUNG_ID, 1 + JONG_ID]
    assert char_ids('댜') == [3 + CHO_ID, 2 + JUNG_ID, JONG_ID]
    assert char_ids('딓') == [3 + CHO_ID, 19 + JUNG_ID, 27 + JONG_ID]


def test_sent_ids():
    expected = [
        [2, 21, 69, 4, 21, 42, 0, 0, 0, 0],
        [4, 21, 42, 2, 21, 69, 0, 0, 0, 0],
    ]
    assert sent_ids('갛나 나갛', maxwords=len(expected)) == expected
