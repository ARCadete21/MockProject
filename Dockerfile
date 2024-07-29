# Use the official Python image as a base image
FROM python:3.11.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose the port Flask is running on
EXPOSE 5001

#Starting the python application
# CMD ["gunicorn", "--bind", "127.0.0.1:5000", "app:app"]
CMD ["flask", "run", "--host=0.0.0.0"]

# # Healthcheck
# HEALTHCHECK CMD curl --fail http://localhost:5000/health || exit 1