 # Dockerfile.train
 FROM python:3.9-slim
 WORKDIR /app
 # Install dependencies
 COPY requirements.txt  .
 COPY params.yaml .
 # assume requirements.txt contains pandas==1.5.3 and scikit-learn==1.2.2
 RUN pip install --no-cache-dir -r requirements.txt
 # Copy the training script
 COPY components/feature-engineering/feature_engineering.py /app/feature_engineering.py
 ENTRYPOINT ["python", "/app/feature_engineering.py"]