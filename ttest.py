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

