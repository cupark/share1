def pre_extract_features(file_name):
    scaler = StandardScaler()
    
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        
        #FFT Fourier Transform for Calculate Power Spectrum
        fft = np.fft.fft(audio)
        signal_spectrum = np.abs(fft)
        
        # Frequency Feature Extract
        #f = np.linspace(0, sample_rate, len(signal_spectrum))
        #half_spectrum = signal_spectrum[:int(len(signal_spectrum)/2)]
        #half_f = f[:int(len(signal_spectrum)/2)]
        
        #stft
        #hop_length = 512
        #n_fft = 2048
        n_fft = int(sample_rate * 0.025)
        hop_length = int(sample_rate * 0.01)
        
        frame_stride = float(hop_length)/sample_rate  #  0.01
        frame_length = float(n_fft)/sample_rate # 0.025
        
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft)
        
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        
        
        #mfccs = librosa.feature.mfcc(audio, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=100)
        #mfccs = librosa.feature.mfcc(audio, sample_rate, log_spectrogram, 100)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, S=log_spectrogram, n_mfcc=175)
        dct_mfccs = dct(x=mfccs, type=2, axis=0)
        scaler = scaler.fit_transform(dct_mfccs)
        pad_width = 0
        
        mfccs_pad = np.pad(scaler, pad_width=((0, 0), (0, pad_width)), mode='constant')
        mfccsscaled = np.mean(scaler.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return scaler, mfccs_pad, mfccsscaled
