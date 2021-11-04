#/usr/bin/env python
from pathlib import Path

import click
import torch
from tqdm import tqdm


@click.command()
@click.option('--input-dir', type=str,
              help='Directory with audios to process.')
@click.option('--output-dir', type=str, default='trimmed',
              help='Directory for audios with pauses trimmed.')
@click.option('--target-sr', type=int, default=48000,
              help='Sample rate of trimmed audios.')
def main(input_dir: str, output_dir: str, target_sr: int) -> None:
    """Remove silence from audios."""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
    )

    (_,  # get_speech_ts
    get_speech_ts_adaptive,
    save_audio,
    read_audio,
    _,  # state_generator
    _,  # single_audio_stream
    collect_chunks) = utils

    path = Path(input_dir)
    processed_path = Path(output_dir)
    processed_path.mkdir(exist_ok=True, parents=True)

    filepath_list = list(path.rglob('*.flac'))
    print(f'Number of audio files found: {len(filepath_list)}')
    print('Performing pausation cutting...')

    log_path = processed_path / "pausation_cutting.log"
    for file in tqdm(filepath_list):
        wave_tensor = read_audio(file, target_sr=target_sr)
        speech_timestamps = get_speech_ts_adaptive(wave_tensor, model)
        speaker_dir = processed_path / file.parent.name
        speaker_dir.mkdir(exist_ok=True)
        try:
            save_audio(
                speaker_dir / file.name,
                collect_chunks(speech_timestamps, wave_tensor),
                target_sr,
            )
        except RuntimeError:
            with open(log_path, "a") as fout:
                fout.write(str(file) + "\n")

    print('Pausation cutting finished.')
    print(f'Trimmed files are located at {output_dir}')

if __name__ == '__main__':
    main()
