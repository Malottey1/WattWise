import requests
import random
import time
import datetime

API_URL = "http://127.0.0.1:5000/api/submit_reading"  # Update if deployed
USER_ID = 1
DEVICE_ID = "simulated_device_001"

def simulate_device():
    while True:
        # Simulate random energy consumption (kWh)
        energy_consumed = round(random.uniform(0.1, 2.5), 3)

        payload = {
            "user_id": USER_ID,
            "device_id": DEVICE_ID,
            "timestamp": datetime.datetime.now().isoformat(),
            "energy_consumed": energy_consumed
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                print(f"[{payload['timestamp']}] Sent: {energy_consumed} kWh ✅")
            else:
                print(f"⚠️ Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Failed to send data: {e}")

        # Send every 10 seconds (adjust as needed)
        time.sleep(10)

if __name__ == "__main__":
    print("Starting IoT Device Simulator... 🚀")
    simulate_device()
