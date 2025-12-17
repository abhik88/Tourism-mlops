
# Use a minimal base image with Python 3.9 installed
FROM python:3.9-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies listed in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Create a non-root user (Required for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Switch to user's home directory
WORKDIR /home/user/app

# Copy application files with correct ownership
COPY --chown=user:user . /home/user/app

# Switch to non-root user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Health check to ensure Streamlit is running
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
