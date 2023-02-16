import queue
import threading
import pyaudio
import numpy as np
import librosa
from sklearn.svm import SVC


# Parameters for audio processing
RATE = 44100  # Sampling rate
CHANNELS = 1  # Number of audio channels
BLOCKSIZE = 1024  # Number of frames per audio block
N_MFCC = 13  # Number of MFCC coefficients to extract
HOP_LENGTH = BLOCKSIZE // 2  # Hop length for the spectrogram

# Parameters for classification
MODEL_PATH = "svm_model.pkl"
CLASSES = ["class1", "class2", "class3"]  # Labels for classification

# Create a queue to hold audio data
data_queue = queue.Queue()

# Load the SVM model
svm_model = SVC(kernel='rbf', gamma='scale')
svm_model = joblib.load(MODEL_PATH)

# Function to process audio data from the queue
def process_data():
    audio_data = np.array([])

    # Read audio data from the queue until the buffer is full
    while len(audio_data) < 2 * RATE:
        try:
            data = data_queue.get(block=False)
            audio_data = np.concatenate((audio_data, data))
        except queue.Empty:
            pass

    # Extract MFCC coefficients from the audio data
    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)

    # Classify the audio using the SVM model
    mfccs = mfccs.reshape(1, -1)  # Reshape to fit the model input
    predicted_class = svm_model.predict(mfccs)[0]

    print("Predicted class:", CLASSES[predicted_class])

# Function to read audio data from the microphone and put it in the queue
def audio_callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    data_queue.put(audio_data)
    return (in_data, pyaudio.paContinue)

# Create an audio stream from the microphone
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paFloat32,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=BLOCKSIZE,
                    stream_callback=audio_callback)

# Start the audio stream
stream.start_stream()

# Process audio data in a separate thread
while True:
    threading.Thread(target=process_data).start()
