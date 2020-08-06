import re
import os
import ast
import json
from jamo import hangul_to_jamo, h2j, j2h, hcj_to_jamo, is_hcj
from jamo.jamo import _jamo_char_to_hcj
import codecs

PAD = '_'
EOS = '~'
PUNC = '!\'(),-.:;?'
SPACE = ' '
_SILENCES = ['sp', 'spn', 'sil']

JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE
ALL_SYMBOLS = list(PAD + EOS + VALID_CHARS) + _SILENCES
s_to_i={c: i for i, c in enumerate(ALL_SYMBOLS)}
#print('s_to_i: ',s_to_i)
KOR_SYMBOLS=ALL_SYMBOLS

Kchar_to_id={c: i for i, c in enumerate(KOR_SYMBOLS)}
id_to_Kchar={i: c for i, c in enumerate(KOR_SYMBOLS)}
