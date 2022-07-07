from pathlib import Path
from shutil import move, rmtree

import click
import numpy as np
from src.constants import SPEAKER_PRINT_DIR


@click.command()
@click.option("--path", type=Path, required=True)
def main(path: Path):
    embs = path.rglob("*npy")
    to_remove = set()
    speaker_emo_ref = dict()
    for emb_path in embs:
        emb_name = emb_path.name
        emb_parts = emb_name.split("_")
        emo = emb_parts[1]
        folder = emb_parts[2]
        emb_name = "_".join(emb_parts[2:])
        save_path = path / folder
        emb = np.load(str(emb_path))
        save_path.mkdir(exist_ok=True, parents=True)
        save_path = save_path / emb_name
        if folder in speaker_emo_ref:
            if emo in speaker_emo_ref:
                speaker_emo_ref[folder][emo]["emb"] += emb
                speaker_emo_ref[folder][emo]["count"] += 1
            else:
                speaker_emo_ref[folder][emo] = {"emb": emb, "count": 1}
        else:
            speaker_emo_ref[folder] = {emo: {"emb": emb, "count": 1}}
        move(emb_path, save_path)
        to_remove.add(emb_path.parent.name)
    for folder in to_remove:
        rmtree(path / folder)
    for speaker in speaker_emo_ref:
        for emo, emo_dict in speaker_emo_ref[speaker].items():
            emb = emo_dict["emb"] / emo_dict["count"]
            save_path = SPEAKER_PRINT_DIR / speaker
            save_path.mkdir(exist_ok=True, parents=True)
            save_path = save_path / f"{emo}.npy"
            np.save(str(save_path), emb)


if __name__ == '__main__':
    main()
