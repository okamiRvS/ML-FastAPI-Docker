# Get the Fast API image with Python version 3.9
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Create the directory for the container
WORKDIR /app
COPY requirements.txt ./requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app.py ./

# Copy the serialized model and the vectors
COPY ./models/test_model.pkl ./models/test_model.pkl

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]