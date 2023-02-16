import pyaudio
import queue
import threading
import time
import numpy as np
import librosa

# 파라미터 설정
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
HOP_LENGTH = 512
N_MFCC = 13
N_FRAMES = 43
BUFFER_LENGTH_SEC = 2
BUFFER_LENGTH = int(RATE * BUFFER_LENGTH_SEC / HOP_LENGTH) * HOP_LENGTH

# 버퍼 큐 초기화
buffer_queue = queue.Queue(maxsize=int(BUFFER_LENGTH/CHUNK))

# 스트림 콜백 함수
def stream_callback(in_data, frame_count, time_info, status):
    # 스트림 데이터를 큐에 추가
    buffer_queue.put(in_data)
    return (in_data, pyaudio.paContinue)

# MFCC 계산 함수
def calculate_mfcc(buffer):
    signal = np.frombuffer(buffer, dtype=np.float32)
    mfccs = librosa.feature.mfcc(signal, sr=RATE, hop_length=HOP_LENGTH, n_mfcc=N_MFCC, n_fft=HOP_LENGTH*N_FRAMES)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    return np.concatenate((mfccs, mfccs_delta, mfccs_delta2))

# MFCC 처리 함수
def process_mfcc(buffer_queue):
    while True:
        # 버퍼 큐에서 2초간의 데이터 가져오기
        buffer = b""
        while buffer_queue.qsize() < int(BUFFER_LENGTH/CHUNK):
            time.sleep(0.1)
        for i in range(int(BUFFER_LENGTH/CHUNK)):
            buffer += buffer_queue.get()
        # MFCC 계산
        mfccs = calculate_mfcc(buffer)
        # 결과 출력
        print(mfccs)

# PyAudio 초기화
p = pyaudio.PyAudio()

# 입력 스트림 열기
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=stream_callback)

# MFCC 처리 스레드 시작
mfcc_thread = threading.Thread(target=process_mfcc, args=(buffer_queue,))
mfcc_thread.daemon = True
mfcc_thread.start()

# 입력 스트림 시작
stream.start_stream()

# 입력 스트림 종료 대기
while stream.is_active():
    time.sleep(0.1)

# PyAudio 종료
stream.stop_stream()
stream.close()
p.terminate()
