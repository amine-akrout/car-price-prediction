# Car Price Prediction 🚗:
## Deploying Machine Learning as Web App on Google Cloud Run using MLflow, flask and github actions
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/amine-akrout/car-price-prediction/Build%20and%20Deploy%20a%20Container?label=Build%20and%20Deploy%20on%20Cloud%20Run)

A machine learning project that predicts car prices using XGBoost. The project includes a complete ML pipeline with model training, monitoring, and a FastAPI web service for real-time predictions.

## 🌟 Features

- End-to-end ML pipeline using Prefect for orchestration
- Model training with XGBoost and MLflow for experiment tracking
- FastAPI web service with HTML interface for predictions
- Docker containerization for both training and serving
- Data drift monitoring with Evidently AI
- Automated testing with pytest
- Google Cloud Storage integration for model artifacts

## 🏗️ Architecture

![Architecture Diagram](./path/to/architecture_diagram.png)

The project is structured into two main components:

1. **Training Pipeline**
   - Data ingestion from Kaggle
   - Data preprocessing and feature engineering
   - Model training with hyperparameter optimization
   - MLflow tracking and model artifact storage

2. **Serving API**
   - FastAPI web service
   - Real-time predictions
   - Data drift monitoring
## 🛠️ Technology Stack

- **ML & Data Science**: Python, pandas, scikit-learn, XGBoost - for data manipulation, model training, and evaluation.
- **MLOps**: MLflow, Prefect, Evidently AI - for experiment tracking, workflow orchestration, and monitoring.
- **Web Framework**: FastAPI - for building the web service and API endpoints.
- **Database**: SQLite - for storing prediction logs and other data.
- **Cloud Storage**: Google Cloud Storage - for storing model artifacts and other files.
- **Containerization**: Docker - for containerizing the application and ensuring consistency across environments.
- **Testing**: pytest - for writing and running tests to ensure code quality.



## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/amine-akrout/car-price-prediction.git
cd car-price-prediction
3. Create a virtual environment:

2. Set up environment variables:

cp .env.example .env
# Edit .env with your configuration
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Training Pipeline

> **Note:** Ensure you have configured your environment variables in the `.env` file before running the training pipeline.


1. Run the training pipeline:
```bash
make train
```

### Serving API

1. Start the API service:
```bash
make serve-api
```

2. Access the web interface at `http://localhost:8000`

### Development Commands

```bash
# Run tests
make test

# Run code quality checks
make quality_checks

# Deploy Prefect workflows
make deploy-prefect

# Stop all services
make stop

# View logs
make logs

# Clean artifacts
make clean
```

## 🧪 Testing

### Setting Up Test Data

Before running the tests, ensure that the test data is set up correctly. You can download the test dataset and place it in the `data/` directory.

1. Download the test dataset:

Run the test suite:
## 📊 Monitoring

Access the monitoring dashboard at `http://localhost:8000/dashboard` to view:
- Data drift metrics
- Model performance statistics
- Prediction logs

> **Note:** Ensure you have configured your environment variables in the `.env` file and started the


## 🔄 MLOps Pipeline

1. Data ingestion from Kaggle
2. Feature engineering and preprocessing
3. Model training with hyperparameter optimization
4. MLflow experiment tracking
5. Model deployment to Google Cloud Storage
6. Serving via FastAPI
7. Continuous monitoring with Evidently AI


## 📂 Project Structure

```bash
├── data/                  # Dataset storage
├── serving/              # API service
│   ├── app.py           # FastAPI application
│   ├── model_loader.py  # Model loading utilities
│   └── monitoring.py    # Drift monitoring
├── training/            # Training pipeline
│   ├── get_data.py     # Data download script
│   └── model_training.py# Training workflow
├── tests/              # Test suite
├── Makefile            # Development commands
└── docker-compose.yml  # Container orchestration
```

## Deploy to cloud Run


```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/carprice  --project=$PROJECT_ID
```

```bash
gcloud run deploy --image gcr.io/$PROJECT_ID/carprice --platform managed  --project=$PROJECT_ID --allow-unauthenticated
```

## CI/CD workflow
Using Github actions and [cloud-run.yml](https://github.com/amine-akrout/car-price-prediction/blob/main/.github/workflows/cloud_run.yml), we could continuously deploy the web app by simply using the term "deploy" in the commit message when pushing to main branch

## 🌐 Demo

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Run tests to ensure your changes do not break the code:
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request


## 🙏 Acknowledgments

* Data can be found here: [kaggle](https://www.kaggle.com/iadelas/car-price-prediction-rf-92/data)
* Federico Tartarini [Continuous deployment to Google Cloud Run using GitHub Actions](https://youtu.be/NCa0RTSUEFQ)
* Michael Harmon [GreenBuildings3: Build & Deploy Models With MLflow & Docker](http://michael-harmon.com/blog/GreenBuildings3.html)
## 📫 Contact

For questions, feedback, or collaboration opportunities, please open an issue in the GitHub repository or reach out via email at [akrout.med.amine@gmail.com](akrout.med.amine@gmail.com).
For questions and feedback, please open an issue in the GitHub repository.
