import pyaudio
import numpy as np
import librosa

# 설정값
CHUNK_SIZE = 1024       # 오디오 스트림에서 한 번에 읽어들일 샘플 수
SAMPLE_RATE = 44100     # 샘플링 주파수
N_MFCC = 20             # 추출할 MFCC 계수의 수
HOP_LENGTH = 512        # MFCC 계산 시 사용할 윈도우 크기

# 버퍼 초기화
buffer = []
buffer_samples = 0

# PyAudio 오디오 스트림 초기화
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

while True:
    # 스트림에서 데이터 읽어들이기
    data = stream.read(CHUNK_SIZE)
    samples = np.frombuffer(data, dtype=np.int16)
    
    # 버퍼에 데이터 추가
    buffer.append(samples)
    buffer_samples += len(samples)
    
    # 2초간 데이터를 쌓으면 MFCC 계산
    if buffer_samples >= SAMPLE_RATE * 2:
        # 버퍼에서 모든 오디오 데이터 추출
        audio = np.concatenate(buffer, axis=0)
        # MFCC 계산
        mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        # 계산 결과 출력
        print("MFCC shape: ", mfccs.shape)
        # 버퍼 초기화
        buffer = []
        buffer_samples = 0

# PyAudio 스트림 닫기
stream.stop_stream()
stream.close()
p.terminate()










import pyaudio

CHUNK_SIZE = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050

audio_stream = pyaudio.PyAudio().open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE
)

while True:
    # 스트림에서 데이터를 읽어들이기
    audio_data = audio_stream.read(CHUNK_SIZE)
    # 입력받은 음성 데이터를 전처리하기
    preprocessed_audio = preprocess_audio_stream(audio_data)
    # SVM 모델에 입력하여 분류하기
    predicted_label = svm_model.predict(preprocessed_audio)
    print(predicted_label)



import librosa

def preprocess_audio_stream(audio_stream):
    # 입력된 음성 데이터를 읽어들이고 MFCC 특징 벡터 추출
    mfccs = librosa.feature.mfcc(y=audio_stream, sr=22050, n_mfcc=40)
    # 데이터를 2차원 배열로 변환
    mfccs = mfccs.T
    return mfccs



def draw_chart_mfccs(mfccs, sample_rate, hop_length):
    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(mfccs, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar()
    plt.title("MFCCS")
    
    
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_plot(frame, mfccs, sample_rate, hop_length):
    plt.cla()  # clear current axis
    librosa.display.specshow(mfccs[:, :frame], sr=sample_rate, hop_length=hop_length, x_axis='time')
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar()
    plt.title("MFCCS")

def animate_mfccs(mfccs, sample_rate, hop_length):
    fig = plt.figure(figsize=FIG_SIZE)
    ani = animation.FuncAnimation(fig, update_plot, fargs=(mfccs, sample_rate, hop_length), frames=mfccs.shape[1], repeat=False)
    plt.show()

