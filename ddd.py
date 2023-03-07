import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import queue

# 오디오 설정
block_size = 2048  # 오디오 블럭 크기
sample_rate = 44100  # 샘플링 주파수
n_mfcc = 13  # MFCC 계수 개수

# MFCC 초기화
mfccs = np.zeros((n_mfcc, 1))

# 큐 초기화
q = queue.Queue()

# 콜백 함수
def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

# Figure 설정
fig, ax = plt.subplots(figsize=(10, 6))

# 애니메이션 함수
def animate(frame):
    ax.clear()
    mfccs = librosa.feature.mfcc(y=q.get(), sr=sample_rate, n_mfcc=n_mfcc)
    librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()

# 오디오 스트림 열기
with sd.InputStream(channels=1, blocksize=block_size,
                    samplerate=sample_rate, callback=audio_callback):
    # 애니메이션 객체 생성
    anim = animation.FuncAnimation(fig, animate, frames=None, interval=20)

    # 애니메이션 실행
    plt.show()
