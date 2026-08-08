# Use lightweight official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose port 8000 for web traffic
EXPOSE 8000

# Command to run FastAPI server on container boot
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]