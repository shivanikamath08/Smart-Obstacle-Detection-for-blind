import requests
import sounddevice as sd
import wave

BOT_TOKEN = "8031980175:AAFS12UC4LkkqtLbtVIsbGTCQKJNmyYOyWA"
CHAT_ID = "6456856712"
AUDIO_FILE = "emergency.ogg"

def record_audio(duration=5, filename="emergency.wav"):
    """ Records audio from the microphone and saves it as a .wav file """
    print("🎙 Recording emergency message...")
    fs = 44100  # Sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype="int16")
    sd.wait()
    print("✅ Recording finished!")

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())

def send_voice_message(file_path):
    """ Sends the recorded voice message to Telegram """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVoice"
    files = {"voice": open(file_path, "rb")}
    data = {"chat_id": CHAT_ID}

    response = requests.post(url, data=data, files=files)
    if response.status_code == 200:
        print("✅ Voice Message Sent Successfully!")
    else:
        print(f"❌ Failed to Send Voice: {response.json()}")

# 🔥 Record and Send
record_audio(duration=5)  # Adjust duration as needed
send_voice_message("emergency.wav")