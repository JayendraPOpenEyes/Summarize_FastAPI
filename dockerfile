# Use an official Python runtime as a parent image
FROM python:3.13-slim

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

# Copy the entire project into the container,
# including your source files, index.html, .env, and Firebase JSON file.
COPY . /app

# Expose the port that the app runs on
EXPOSE 8000

# Run the FastAPI application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
