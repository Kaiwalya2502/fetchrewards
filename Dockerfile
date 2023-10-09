# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && pip install --trusted-host pypi.python.org -r requirements.txt \
    && mkdir -p data \
    && mkdir -p plots \
    && mkdir -p models \
    && wget -O data/data_daily.csv https://fetch-hiring.s3.amazonaws.com/machine-learning-engineer/receipt-count-prediction/data_daily.csv \
    && apt-get purge -y --auto-remove wget \
    && rm -rf /var/lib/apt/lists/*
# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run train.py when the container launches
RUN python train.py data/data_daily.csv

# Command to run the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app.app:app"]
