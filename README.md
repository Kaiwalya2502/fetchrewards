
# Receipt Count Forecasting Application

## Overview
The Receipt Count Forecasting Application is a web-based solution that provides visual representation and future predictions of receipt count data using machine learning models and interactive visualizations.

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/)

## Directory Structure
```
/fetch
    /app
        /static
            /js
        /templates
    /models
    /data
    /plots
```

## Key Components
- `/fetch/app`: Contains main application logic and UI.
    - `app.py`: Main application script.
    - `/static/js`: Contains JavaScript files like `plotly.min.js`.
    - `/templates`: Contains HTML and CSS files defining the UI.
- `/fetch/models`: Stores machine learning models like `lr_model.pkl`.
- `/fetch/data`: Contains datasets like `data_daily.csv`.
- `/fetch/plots`: Contains generated plots.

## Additional Files
- `Dockerfile`: Instructions to build the Docker image.
- `train.py`: Script to train the model and save it.
- `requirements.txt`: Lists Python packages required to run the app.

## How to Use

### Build and Run Docker Container Locally
1. **Build Docker Image**: Navigate to the directory containing the Dockerfile and execute:
   ```sh
   docker build -t fetch .
   ```
2. **Run Docker Container**:
   ```sh
   docker run -p 4000:80 fetch
   ```
3. Access the app at [http://localhost:4000](http://localhost:4000).

### Pull from Docker Hub (If Available)
1. **Pull Docker Image**:
   ```sh
   docker pull [your-dockerhub-username]/myflaskapp
   ```
2. **Run Docker Container**:
   ```sh
   docker run -p 4000:80 [your-dockerhub-username]/fetch
   ```
3. Access the app at [http://localhost:4000](http://localhost:4000).

## Interaction with the App
- Visualize past receipt count data and predict future values interactively.

## Conclusion
The app provides an interactive platform to visualize and predict receipt counts, leveraging machine learning for informed decision-making.
