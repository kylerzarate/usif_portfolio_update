FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the dependencies file
COPY requirements.txt ./

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the scripts directory into the container
COPY . /scripts /app/scripts

# Copy all remaining files
COPY . /app/

# Define the default command to run the main script
CMD ["python", "scripts/main.py"]
