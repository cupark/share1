import pyaudio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# PyAudio 객체를 생성합니다.
p = pyaudio.PyAudio()

# 오디오 스트림을 엽니다.
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True)

# 그래프를 초기화합니다.
fig, ax = plt.subplots()
spec = ax.imshow(np.zeros((128, 128)), cmap='viridis', origin='lower', aspect='auto')

# 스펙트로그램을 계산합니다.
def update(frame):
    y = np.frombuffer(stream.read(1024), dtype=np.float32)
    spec.set_data(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=44100), ref=np.max))
    return spec

# 애니메이션을 생성합니다.
ani = FuncAnimation(fig, update, interval=50)

# 애니메이션을 화면에 출력합니다.
plt.show()

# 오디오 스트림을 닫습니다.
stream.stop_stream()
stream.close()
p.terminate()
