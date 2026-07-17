# python image
FROM python:3.12-slim
# set working directory
WORKDIR /workdir
# copy dependencies
COPY requirements-lock.txt ./
# install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# copy script -- add later

# set CMD -- add later
