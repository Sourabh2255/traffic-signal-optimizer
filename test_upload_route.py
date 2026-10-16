import os
import requests

BASE_URL = "http://127.0.0.1:5000"

img_dir = r"data\emergency_yolo\test\images"
images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")][:4]
wav_path = r"data\audio\no_siren\no_siren_000.wav"

print(f"Testing with images: {images}")
print(f"Testing with audio: {wav_path}")

files = {
    "north": ("north.jpg", open(images[0], "rb"), "image/jpeg"),
    "east": ("east.jpg", open(images[1], "rb"), "image/jpeg"),
    "south": ("south.jpg", open(images[2], "rb"), "image/jpeg"),
    "west": ("west.jpg", open(images[3], "rb"), "image/jpeg"),
    "audio_south": ("siren.wav", open(wav_path, "rb"), "audio/wav"),
}

data = {
    "emergency_north": "Auto-Detect",
    "emergency_east": "Auto-Detect",
    "emergency_south": "Auto-Detect",
    "emergency_west": "Auto-Detect",
}

session = requests.Session()
response = session.post(f"{BASE_URL}/upload", files=files, data=data, allow_redirects=True)

print("Response status code:", response.status_code)
print("Response final URL:", response.url)
assert response.status_code == 200, f"Expected 200, got {response.status_code}"
assert "/analysis" in response.url or "Analysis" in response.text or "Cycle" in response.text, "Analysis route not reached"

if "ModuleNotFoundError" in response.text:
    print("FAILED: ModuleNotFoundError still present!")
    exit(1)
elif "Traceback" in response.text:
    print("FAILED: Traceback found in response!")
    exit(1)
else:
    print("SUCCESS: Upload and analysis workflow completed without errors!")
