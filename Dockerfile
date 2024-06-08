FROM python:3.10.4:alpine3.20

WORKDIR /app
COPY . ./

RUN pip install --upgrade pip
RUN pip install -r ./deploy-google/requirements.txt

CMD ["sh", "-c", "cd deploy-google && python3 -m flask run --host=0.0.0.0"]