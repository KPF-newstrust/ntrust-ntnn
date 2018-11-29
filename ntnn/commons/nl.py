import re
import math
from ntnn.utils import arr_from

CHO, JUNG, JONG = 19, 21, 28

UNK = '\u00e0'  # unknown chars. should be lowercase
ASCII = range(0x20, 0x7e+1)
HANGUL = range(0xac00, 0xd7a3+1)
SYMBOL = range(0x2000, 0x206f+1)
SYMBOL2 = range(0x318d, 0x318d+1)

PAD_ID = 0
UNK_ID = PAD_ID + 1
ASCII_ID = UNK_ID + 1
HANGUL_ID = ASCII_ID + 1
CHO_ID = HANGUL_ID
JUNG_ID = CHO_ID + CHO
JONG_ID = JUNG_ID + JUNG
SYMBOL_ID = JONG_ID + JONG
SYMBOL2_ID = SYMBOL_ID + len(SYMBOL)


def char_ids(ch):
    n = ord(ch)
    if n in HANGUL:
        han = n - HANGUL[0]
        cho = int(han / JONG / JUNG) + HANGUL_ID
        jung = int(han / JONG) % JUNG + JUNG_ID
        jong = int(han % JONG) + JONG_ID
        return [cho, jung, jong]

    if n in ASCII:
        return n - ASCII[0] + ASCII_ID

    if n in SYMBOL:
        return n - SYMBOL[0] + SYMBOL_ID

    if n in SYMBOL2:
        return n - SYMBOL2[0] + SYMBOL2_ID
    return [UNK_ID]


def sent_ids(sentence, **kwargs):
    return words_ids(re.split(r'\s', sentence), **kwargs)


def words_ids(sentence, maxwords=100, maxchars=10):
    """
    Args:
        sentence: [nwords]
        maxwords: max words per sentence
        maxchars: max chars per word
    Return: [maxwords, maxchars]
    """

    word_ids = []
    for word in sentence:
        word_id = []
        for ch in word:
            word_id += char_ids(ch)
        word_ids.append(arr_from(word_id, maxchars, PAD_ID))

    pad_word = arr_from([], maxchars, PAD_ID)
    return arr_from(word_ids, maxwords, pad_word)
