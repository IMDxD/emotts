FROM python:3.8-slim

WORKDIR /emotts

# Needed to run "RUN source ~/.bashrc" later
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && apt-get install -y wget
RUN wget --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3 -f

RUN miniconda3/bin/conda init bash

# Reload .bashrc
RUN source ~/.bashrc

COPY environment.yml .
RUN conda env create -f environment.yml

COPY run.sh .

# CMD [ "bash", "-i", "run.sh" ]