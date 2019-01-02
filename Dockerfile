M python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

# Install pytorch and fastai
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN pip install fastai

# Install starlette and uvicorn
RUN pip install starlette uvicorn python-multipart aiohttp

ADD pet-prophet.py pet-prophet.py
ADD pets-model.pth pets-model.pth

# Run it once to trigger resnet download
RUN python pet-prophet.py

EXPOSE 8008

# Start the server
CMD ["python", "pet-prophet.py", "serve"]

