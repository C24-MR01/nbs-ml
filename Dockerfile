FROM python:3.12.2

WORKDIR /app
COPY . ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["sh", "-c", "cd deploy-google && python3 -m flask run --host=0.0.0.0"]