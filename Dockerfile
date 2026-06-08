# Create the base image
FROM python:3.13.16-slim
# Make the working directory
WORKDIR /app
# Copy the requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]