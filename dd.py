import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# wav 파일 로드
audio_file = 'audio.wav'
y, sr = librosa.load(audio_file, sr=None, mono=True, duration=2)

# mfcc 추출
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# figure 설정
fig, ax = plt.subplots(figsize=(10, 6))
ax.set(xlim=(0, mfccs.shape[1]), ylim=(-100, 100))
plt.xlabel("Frame")
plt.ylabel("MFCC Coefficients")

# 그래프 초기화
line, = ax.plot([], [], lw=2)

# 애니메이션 함수
def animate(frame):
    x = np.arange(0, frame+1)
    y = mfccs[:, :frame+1]
    line.set_data(x, y)
    return line,

# 애니메이션 객체 생성
anim = animation.FuncAnimation(fig, animate, frames=mfccs.shape[1], interval=20)

# 애니메이션 실행
plt.show()
