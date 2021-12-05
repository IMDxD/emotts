# MSP Improv Preprocessing

## Prerequisites

* Docker: 20.10.7

## Usage

```bash
docker build --rm --tag emotts ./msp_improv_preprocessing
docker run --rm -it -v $(pwd):/emotts/repo emotts
```

Processed data will be located at `data/processed/mfa_outputs` (as `.TextGrid`, grouped by speaker IDs) and `data/processed/mels` (as `.pkl`, grouped by speaker IDs).