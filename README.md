# Deploying Machine Learning as Web App on Google Cloud Run using MLflow, flask and github actions ![GitHub Workflow Status](https://img.shields.io/github/workflow/status/amine-akrout/car-price-prediction/Build%20and%20Deploy%20a%20Container?label=Build%20and%20Deploy%20on%20Cloud%20Run)

## Requirements
* Python 3.8
* Docker
* Google Cloud Plateform account

## Quick Start
* Clone the repository
<pre>
git clone https://github.com/amine-akrout/car-price-prediction
</pre>
* Create a virtual and install requirements
<pre>
python -m venv
pip install -r requirements.txt
</pre>
* Train XGBoostmodel and log metrics and artifacts with MLflow
<pre>
python ./model.py
</pre>

## Test locally
To test the web app locally using docker, start by building the image from the Dockerfile
<pre>
docker build --pull --rm -f "Dockerfile" -t carprice:latest "."
</pre>

<pre>
docker run --rm -d  carprice:latest
</pre>

## Deploy to cloud Run
<pre>
</pre>

## Demo
<pre>
</pre>