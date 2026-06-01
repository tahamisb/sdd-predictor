FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]