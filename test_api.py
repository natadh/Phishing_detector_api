import requests

url = "http://127.0.0.1:8000/predict/"
data = {"email_text": "Your account has been suspended. Click here to verify your details."}
response = requests.post(url, json=data)
print(response.json())
