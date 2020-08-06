""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''


#import cmudict
#from korean import KOR_SYMBOLS
from .korean import KOR_SYMBOLS

kor_symbols=KOR_SYMBOLS
symbols=kor_symbols
