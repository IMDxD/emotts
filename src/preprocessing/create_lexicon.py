"""
Util to create phonematic lexicon for given corpus.
Espeak backend must be installed on the system.
"""

from pathlib import Path

import click
from phonemizer.phonemize import phonemize
from phonemizer.separator import Separator

ENCODING = "utf8"
SEPARATOR = Separator(phone=" ", syllable="", word="\n")
CORPUS_STRING_WORD_SEPARATOR = " "


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=Path,
    help="Path to corpus file (1 word per line).",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=Path,
    default="russian-lexicon-espeak.txt",
    help="Filepath to write lexicon.",
    required=False,
)
@click.option(
    "-l",
    "--language",
    type=str,
    default="en-us",
    help="Language in espeak format. See for more info: https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md",
    required=False,
)
def main(input_path: Path, output_path: Path, language: str) -> None:

    print("Reading corpus...", end=" ")
    with open(input_path, "r", encoding=ENCODING) as corpus_file:
        corpus_string = CORPUS_STRING_WORD_SEPARATOR.join(
            corpus_file.read().splitlines()
        )
    print("Done.")
    print(corpus_string[:80], end="\n\n")

    print("Getting phonemizations...", end=" ")
    phones_string = phonemize(
        corpus_string,
        language=language,
        backend="espeak",
        preserve_punctuation=True,
        with_stress=True,
        strip=True,
        separator=SEPARATOR,
    )
    print("Done.")
    print(phones_string[:80], end="\n\n")

    print("Creating lexicon...", end=" ")
    corpus = corpus_string.split(CORPUS_STRING_WORD_SEPARATOR)
    phones = phones_string.split(SEPARATOR.word)
    assert len(corpus) == len(phones), (
        f"# of words should match # of phones after phomenization\n"
        f"but you have {len(corpus)} words and {len(phones)} phones\n"
        f"First 5 words: {corpus[:5]}\n"
        f"First 5 phones: {phones[:5]}\n"
    )
    lexicon = [f"{word}\t{phon}" for word, phon in zip(corpus, phones)]
    lexicon_str = "\n".join(lexicon)
    print("Done.", end="\n\n")

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding=ENCODING) as lexicon_file:
        lexicon_file.write(lexicon_str)
    print(f"Lexicon file saved at:\n{output_path}", end="\n\n")


if __name__ == "__main__":
    main()
