import shutil
from pathlib import Path

import click
import pandas as pd

from src.constants import REFERENCE_PATH

DEFAULT_BACKUP_PATH = REFERENCE_PATH.with_name(REFERENCE_PATH.name + "_old")
ENCODING = "utf8"


@click.command()
@click.option("-r", "--reference-dir", type=click.Path(exists=True), required=False, default=REFERENCE_PATH)
@click.option("-m", "--mels-dir", type=click.Path(exists=True))
@click.option("-f", "--new-refs-csv", type=click.Path(exists=True))
@click.option("-b", "--backup-dir", type=click.Path(), required=False, default=DEFAULT_BACKUP_PATH)
@click.option("-e", "--reference-extension", type=click.STRING, required=False, default="pkl")
@click.option("-E", "--mels-extension", type=click.STRING, required=False, default="pth")
def update_refs_from_csv(
        reference_dir: Path,
        mels_dir: Path,
        backup_dir: Path,
        new_refs_csv: Path,
        reference_extension: str,
        mels_extension: str
) -> None:

    # Backup
    shutil.rmtree(backup_dir, ignore_errors=True)
    shutil.copytree(reference_dir, backup_dir)

    df = pd.read_csv(new_refs_csv, converters={'speaker_id': lambda x: str(x)})
    old_refs = list(reference_dir.rglob("*." + reference_extension))

    for ref in old_refs:

        speaker = ref.parent.name
        emo = ref.stem

        new_ref_df = df[(df.emotion == emo) & (df.speaker_id == speaker)]
        assert len(new_ref_df) == 1

        new_ref_wavid = new_ref_df.wav_id.values[0]
        new_ref_path = (Path(mels_dir) / speaker / new_ref_wavid).with_suffix("." + mels_extension)
        print(f"REPLACING:\t{ref}\nWITH:\t{new_ref_path}\n")
        assert new_ref_path.exists()

        shutil.copy2(new_ref_path, ref)

    print("BACKUP SAVED TO:", backup_dir)


if __name__ == "__main__":
    update_refs_from_csv()
