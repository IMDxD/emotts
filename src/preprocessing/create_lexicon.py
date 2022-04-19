"""
Util to create phonematic lexicon for given corpus.
Espeak backend must be installed on the system.
"""

from pathlib import Path

import click
from phonemizer.phonemize import phonemize
from phonemizer.separator import Separator
from tqdm import tqdm

ENCODING = "utf8"
SEPARATOR = Separator(phone=" ", syllable="", word="")


@click.command()
@click.option(
    "--input-path", type=Path, help="", required=True,
)
@click.option(
    "--output-path",
    type=Path,
    default="russian-lexicon-espeak.txt",
    help="Filepath to write lexicon.",
    required=True,
)
def main(input_path: Path, output_path: Path) -> None:
    with open(input_path, "r", encoding=ENCODING) as corpus_file:
        with open(output_path, "w", encoding=ENCODING) as lexicon_file:
            for word in tqdm(list(corpus_file.readlines())):
                word = word.strip()
                phones = phonemize(
                    word,
                    language="ru",
                    backend="espeak",
                    preserve_punctuation=True,
                    with_stress=True,
                    separator=SEPARATOR,
                )
                lexicon_line = f"{word}\t{phones}\n"
                lexicon_file.write(lexicon_line)
    print(f"Lexicon file saved at:\n{output_path}")


if __name__ == "__main__":
    main()
