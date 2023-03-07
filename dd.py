import librosa
import librosa.display
import matplotlib.pyplot as plt

# wav 파일 로드
audio_file = 'audio.wav'
y, sr = librosa.load(audio_file, sr=None, mono=True)

# mfcc 추출
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# MFCC 그래프 그리기
plt.figure(figsize=(10, 6))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.tight_layout()
plt.show()
