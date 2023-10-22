# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system packages if needed
# For example, if your application requires any system libraries, install them here using apt-get or yum

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the application runs on
EXPOSE 5000

# Run the application
CMD ["python", "task1.py"]
