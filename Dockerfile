FROM continuumio/miniconda3

WORKDIR /emotts

COPY . .

# Create conda environment from config file
RUN conda env create -f environment.yml

# Start web-server
EXPOSE 8080
ENTRYPOINT ["conda", "run", "-n", "emotts", \
            "streamlit", "run", "app.py", \
            "--server.port", "8080"]