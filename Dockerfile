FROM python:3.9 
# Or any preferred Python version.
ADD app.py .

CMD [“python”, “./app.py”] 
# Or enter the name of your unique directory and parameter set.