# Use an official Python runtime as a parent image
FROM python:3.8
# docker login final.azurecr.io -u Final
# docker tag final_image:tag final.azurecr.io/final_image:tag
# docker push final.azurecr.io/final_image:tag
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt  # Adjust based on your actual requirements file name

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]


