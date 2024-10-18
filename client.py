
import requests

URL = "http://127.0.0.1:5000/predict"
TEST_AUDIO_FILE_PATH = "test/left.wav"

if __name__ == "__main__":
	
	audio_file = open(TEST_AUDIO_FILE_PATH,"rb")
	values = {"file" : (TEST_AUDIO_FILE_PATH,audio_file,"audio/wav")}
	response = requests.post(URL,files=values)

	print(response.status_code) # Check if the server returns 200
	print(response.text) # Inspect the raw response content

	data = response.json()

	print(f"Predicted keyword is: {data['keyword']}")


