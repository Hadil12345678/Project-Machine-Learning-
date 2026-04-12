import requests
import streamlit as st
import os
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000")
class APIClient:
    def __init__(self):
        self.base = FASTAPI_URL
        self.token = None

    # ✅ health check
    def is_alive(self):
        try:
            r = requests.get(f"{self.base}/health", timeout=3)
            return r.status_code == 200
        except:
            return False

    # ✅ login
    def login(self, username: str, password: str):
        response = requests.post(
            f"{self.base}/login",
            data={
                "username": username,
                "password": password
            }
        )

        if response.status_code == 200:
            data = response.json()
            self.token = data["access_token"]
            return True
        return False

    # ✅ predict
    def predict(self, patient_data: dict):
        if not self.token:
            st.error("⚠️ You must login first")
            return None

        headers = {
            "Authorization": f"Bearer {self.token}"
        }

        payload = {
            "Age": patient_data["age"],
            "Num of pregnancies": patient_data["pregnancies"],
            "Smokes": patient_data["smokes"],
            "Hormonal Contraceptives": patient_data["hormonal"]
        }

        response = requests.post(
            f"{self.base}/predict",
            json=payload,
            headers=headers
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(response.text)
            return None