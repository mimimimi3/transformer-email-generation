# python image
FROM python:3.10.12
# set working directory
WORKDIR /workdir
# copy dependencies
COPY requirements-lock.txt ./
# install dependencies
RUN pip install --no-cache-dir -r requirements-lock.txt
# copy configs folder
COPY configs ./configs
# copy source code folder
COPY src ./src
# run the command
CMD ["python", "--version"]
