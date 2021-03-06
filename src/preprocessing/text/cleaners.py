"""
From https://github.com/keithito/tacotron

Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""
import re

from unidecode import unidecode

from .numbers import normalize_numbers

# Regular expression matching alphanumeric + cyrillic characters and whitespace
RUSSIAN_NON_LETTERS_PATTERN = "[^A-Za-zА-я0-9 ]+"

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile(f"\\b{x[0]}(?=[ ])", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text: str) -> str:
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text: str) -> str:
    return normalize_numbers(text)


def lowercase(text: str) -> str:
    return text.lower()


def collapse_whitespace(text: str) -> str:
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text: str) -> str:
    return unidecode(text)


def basic_cleaners(text: str) -> str:
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text: str) -> str:
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text: str) -> str:
    """Pipeline for English text, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def punctuation_to_whitespace(text: str) -> str:
    pattern = RUSSIAN_NON_LETTERS_PATTERN
    text = re.sub(pattern, " ", text)
    return text


def russian_cleaners(text: str) -> str:
    """Pipeline for Russian text, cleaning only."""
    text = lowercase(text)
    text = punctuation_to_whitespace(text)
    text = collapse_whitespace(text)
    return text
