# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies required for PDF, image, and OCR processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    tesseract-ocr \
    wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy the entire project into the container
COPY . /app

# Validate environment variables (optional but recommended)
RUN python -c "import os, json; \
    assert os.environ.get('FIREBASE_CONFIG'), 'FIREBASE_CONFIG not set'; \
    json.loads(os.environ['FIREBASE_CONFIG']); \
    print('Firebase config validation successful')"

# Expose the port that the app runs on
EXPOSE 8000

# Run the FastAPI application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]