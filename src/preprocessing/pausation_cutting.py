#!/usr/bin/env python
from pathlib import Path
import ssl

import click
import torch
from tqdm import tqdm


@click.command()
@click.option("--input-dir", type=str,
              help="Directory with audios to process.")
@click.option("--output-dir", type=str, default="trimmed",
              help="Directory for audios with pauses trimmed.")
@click.option("--target-sr", type=int, default=48000,
              help="Sample rate of trimmed audios.")
@click.option("--audio-ext", type=str, default="flac",
              help="Extension of audio files.")
def main(input_dir: str, output_dir: str, audio_ext: str, target_sr: int) -> None:
    """Remove silence from audios."""
    
    # Disables SSL cert check for urllib which will be called
    # in subsequent call of torch.hub.load
    ssl._create_default_https_context = ssl._create_unverified_context
    
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
    )

    (
        _,  # get_speech_ts
        get_speech_ts_adaptive,
        save_audio,
        read_audio,
        _,  # state_generator
        _,  # single_audio_stream
        collect_chunks
    ) = utils

    path = Path(input_dir)
    processed_path = Path(output_dir)
    processed_path.mkdir(exist_ok=True, parents=True)

    filepath_list = list(path.rglob(f"*.{audio_ext}"))
    print(f"Number of audio files found: {len(filepath_list)}")
    print("Performing pausation cutting...")

    log_path = processed_path / "pausation_cutting.log"
    for filepath in tqdm(filepath_list):
        wave_tensor = read_audio(filepath, target_sr=target_sr)
        speech_timestamps = get_speech_ts_adaptive(wave_tensor, model)
        speaker_dir = processed_path / filepath.parent.name
        speaker_dir.mkdir(exist_ok=True)
        try:
            save_audio(
                speaker_dir / filepath.name,
                collect_chunks(speech_timestamps, wave_tensor),
                target_sr,
            )
        except RuntimeError:
            with open(log_path, "a") as fout:
                fout.write(str(filepath) + "\n")

    print("Pausation cutting finished.")
    print(f"Trimmed files are located at {output_dir}")


if __name__ == "__main__":
    main()
