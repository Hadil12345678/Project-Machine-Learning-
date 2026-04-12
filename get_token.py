import requests

response = requests.post(
    "http://127.0.0.1:8000/login",
    data={
        "username": "test",
        "password": "1234"
    }
)

print(response.json())