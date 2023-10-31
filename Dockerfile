# Use the Python 3.9.17 image as the base
FROM python:3.9.17

# Create a directory inside the container
RUN mkdir /exp

# Set the working directory to /exp
WORKDIR /exp

# Copy the contents of your local directory into the container's /exp directory
COPY . /exp/

# Install the Python dependencies from requirements.txt
RUN pip install -r /exp/requirements.txt

