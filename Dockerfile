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
# copy shared utility helpers
COPY utils ./utils
COPY pyproject.toml ./
# install the local utils package so imports work without PYTHONPATH hacks
RUN pip install --no-cache-dir -e .
# run the command
CMD ["python", "--version"]
