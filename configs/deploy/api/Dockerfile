FROM continuumio/miniconda3

WORKDIR /emotts

COPY . .

# Create conda environment from config file
RUN conda env create -f environment.yml

# Update models
RUN ["conda", "run", "-n", "emotts", \
    "bash", "models/download_hifigan.sh"]
RUN ["conda", "run", "-n", "emotts", \
    "bash", "models/download_tacotron.sh"]

# Start web-server
EXPOSE 8080
ENTRYPOINT ["conda", "run", "-n", "emotts", \
            "uvicorn", "api:app", \
            "--host", "0.0.0.0", \
            "--port", "8080"]