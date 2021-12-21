# Based on the original HiFi-GAN [paper](https://arxiv.org/abs/2010.05646) and [repo](https://github.com/jik876/hifi-gan): 
HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.
Jungil Kong, Jaehyeon Kim, Jaekyoung Bae


## Preprocessing for flac files

VCTK dataset has flac format which needs to be transformed to wav and then downsampled to ~22kHz:
```
python preprocess_flac.py --input_dir="/input_dir/" --output_dir="/output_dir/" --sample_rate=22050
```
Then one can use ```inference.py``` from below to compare the original wav with the generated one.


## Fine-tuning

We use a pretrained universal model to perform fine-tuning.<br/>
[Download universal model](https://drive.google.com/drive/folders/1YuOoV3lO2-Hhn1F2HJ2aQ4S0LC1JdKLd)<br/> 
First, we need mel-spectrogams generated from out feature-extractor model with teacher-forcing.<br/>
Then, we put them into `mels` and original audios to `wavs` by default.<br/>
Finally, run:
    ```
    python train.py --fine_tuning True --config config.json
    ```


## Inference for end-to-end speech synthesis
1. Make `test_mel_files` directory and copy generated mel-spectrogram files into the directory.<br>
You can generate mel-spectrograms using [Tacotron2](https://github.com/NVIDIA/tacotron2), 
[Glow-TTS](https://github.com/jaywalnut310/glow-tts) and so forth.
2. Run the following command.
    ```
    python inference_e2e.py --checkpoint_file [generator checkpoint file path]
    ```
Generated wav files are saved in `generated_files_from_mel` by default.<br>
You can change the path by adding `--output_dir` option.



