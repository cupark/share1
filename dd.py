import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# wav 파일 로드
audio_file = 'audio.wav'
y, sr = librosa.load(audio_file, sr=None, mono=True)

# mfcc 추출
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# figure 설정
fig, ax = plt.subplots(figsize=(10, 6))

# 애니메이션 함수
def animate(frame):
    ax.clear()
    librosa.display.specshow(mfccs[:, :frame+1], x_axis='time', ax=ax)
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()

# 애니메이션 객체 생성
anim = animation.FuncAnimation(fig, animate, frames=mfccs.shape[1], interval=20)

# 애니메이션 실행
plt.show()
