import pyaudio
import numpy as np
import librosa
import joblib
from sklearn import svm

# SVM 모델 로드
svm_model = joblib.load("svm_model.pkl")

# pyaudio 객체 생성
CHUNK = 1024
RATE = 8000
p = pyaudio.PyAudio()

# 마이크로부터 데이터 스트림 받아오기
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

# 데이터 분류
while True:
    # 스트림에서 데이터 읽어오기
    data = stream.read(CHUNK)
    # 읽어온 데이터를 numpy 배열로 변환
    samples = np.frombuffer(data, dtype=np.int16)
    # MFCC 특징 벡터 추출
    mfcc = librosa.feature.mfcc(samples.astype(float), sr=RATE)
    # SVM 모델을 사용하여 분류
    label = svm_model.predict(mfcc.T)
    # 분류 결과 출력
    print(label)
