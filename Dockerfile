FROM python:3.11

WORKDIR /app
COPY . ./

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["sh", "-c", "cd deploy-google && python3 -m flask run --host=0.0.0.0"]