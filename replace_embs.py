from pathlib import Path
from shutil import move, rmtree

import click


@click.command()
@click.option("--path", type=Path, required=True)
def main(path: Path):
    embs = path.rglob("*npy")
    to_remove = set()
    for emb_path in embs:
        emb_name = emb_path.name
        emb_parts = emb_name.split("_")
        folder = emb_parts[2]
        emb_name = "_".join(emb_parts[2:])
        save_path = path / folder
        save_path.mkdir(exist_ok=True, parents=True)
        save_path = save_path / emb_name
        move(emb_path, save_path)
        to_remove.add(emb_path.parent.name)
    for folder in to_remove:
        rmtree(path / folder)


if __name__ == '__main__':
    main()
