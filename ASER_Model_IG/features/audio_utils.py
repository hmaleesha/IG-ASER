import librosa
import numpy as np

from utils import downsample_with_max_pooling
import noisereduce as nr


NUM_MFCC = 128


def load_wav(filename: str, sample_rate: int):
    return librosa.load(filename, sample_rate)


def add_missing_padding(audio, sr, duration):
    signal_length = duration * sr
    audio_length = audio.shape[0]
    padding_length = signal_length - audio_length
    if padding_length > 0:
        padding = np.zeros(padding_length)
        signal = np.hstack((audio, padding))
        return signal
    return audio


def split_audio(signal, sr, split_duration):
    length = split_duration * sr

    if length < len(signal):
        frames = librosa.util.frame(signal, frame_length=length, hop_length=length).T
        return frames
    else:
        audio = add_missing_padding(signal, sr, split_duration)
        frames = [audio]
        return np.array(frames)


def spectrogram_from_audio_file(path, sr, duration):
    wav, sr = librosa.load(path, sr=sr)
    audio_segment = split_audio(wav, sr, duration)

    # Use values from Sun H
    stft = np.abs(librosa.stft(audio_segment[0], n_fft=278, hop_length=229))
    downsampled = downsample_with_max_pooling(stft)

    # # get Fast Fourier Transformation
    # # Sun H. et al EmotionNAS, 2022
    # hamming_window_size = int((25 / 1000) * sr)  # 25ms Window length
    # hop_length = int((25 - 14) / 1000 * sr)  # 14ms overlap
    #
    # frame = librosa.util.frame(audio_segment[0], hamming_window_size, hop_length)
    # hamming_w = librosa.filters.get_window("hamming", hamming_window_size)
    # wind_frames = hamming_w.reshape(-1, 1) * frame
    # sp = np.real(fft2(wind_frames, (128, 128)))
    return downsampled

    # return librosa.feature.melspectrogram(y=frames[0], sr=sr, hop_length=506)


def mfcc_from_audio_file(path, sr, duration):
    wav, sr = librosa.load(path, sr=sr)
    
    #D = librosa.stft(wav)
    #magnitude, phase = np.abs(D), np.angle(D)

    # Estimate noise profile from quieter parts of the audio
    #noise_magnitude = np.percentile(magnitude, 10, axis=1)

    # Spectral gating: Reduce noise from the signal
    #reduced_noise_magnitude = np.maximum(magnitude - noise_magnitude[:, np.newaxis], 0)
    #D_cleaned = reduced_noise_magnitude * np.exp(1j * phase)

    # Inverse STFT to get the cleaned audio
    #y_cleaned = librosa.istft(D_cleaned)

    # Scale cleaned audio to 16-bit PCM range and convert to int16
    #y_cleaned_scaled = np.int16(y_cleaned / np.max(np.abs(y_cleaned)) * 32750)
    #gated_audio = np.where(np.abs(wav) > 0.02, wav, 0)
    #audio_segment = split_audio(gated_audio, sr, duration)
    #reduced_audio = nr.reduce_noise(y=wav, sr=sr)
    audio_segment = split_audio(wav, sr, duration)

    mfcc = librosa.feature.mfcc(y=audio_segment[0], sr=sr, n_mfcc=NUM_MFCC)

    downsampled = downsample_with_max_pooling(mfcc, (1, 4))
    #print(downsampled.shape)
    return downsampled

def chroma_from_audio_file(path, sr, duration):
    wav, sr = librosa.load(path, sr=sr)
    audio_segment = split_audio(wav, sr, duration)

    # Extract chroma features
    chroma = librosa.feature.chroma_stft(y=audio_segment[0], sr=sr, n_chroma=12)
    
    # Optionally apply downsampling
    downsampled_chroma = downsample_with_max_pooling(chroma, (1, 4))

    return downsampled_chroma


def tonnetz_from_audio_file(path, sr, duration):
    wav, sr = librosa.load(path, sr=sr, duration=duration)
    audio_segment = split_audio(wav, sr, duration)
    chroma = librosa.feature.chroma_stft(y=audio_segment[0], sr=sr, n_chroma=12)
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    downsampled_tonnetz = downsample_with_max_pooling(tonnetz, (1, 4))
    return downsampled_tonnetz

def tonnetzchroma__from_audio(path, sr, duration):
    wav, sr = librosa.load(path, sr=sr, duration=duration)
    audio_segment = split_audio(wav, sr, duration)
    chroma = librosa.feature.chroma_stft(y=audio_segment[0], sr=sr, n_chroma=12)
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
   
    chroma = downsample_with_max_pooling(chroma, (1, 4))
    tonnetz = downsample_with_max_pooling(tonnetz, (1, 4))
    return chroma, tonnetz

'''def pitch_and_intensity_from_audio(path, sr, duration):
    wav, sr = librosa.load(path, sr=sr)
    
    #D = librosa.stft(wav)
    #magnitude, phase = np.abs(D), np.angle(D)

    # Estimate noise profile from quieter parts of the audio
    #noise_magnitude = np.percentile(magnitude, 10, axis=1)

    # Spectral gating: Reduce noise from the signal
    #reduced_noise_magnitude = np.maximum(magnitude - noise_magnitude[:, np.newaxis], 0)
    #D_cleaned = reduced_noise_magnitude * np.exp(1j * phase)

    # Inverse STFT to get the cleaned audio
    #y_cleaned = librosa.istft(D_cleaned)

    # Scale cleaned audio to 16-bit PCM range and convert to int16
    #y_cleaned_scaled = np.int16(y_cleaned / np.max(np.abs(y_cleaned)) * 32750)
    #gated_audio = np.where(np.abs(wav) > 0.02, wav, 0)
    #audio_segment = split_audio(gated_audio, sr, duration)
    #reduced_audio = nr.reduce_noise(y=wav, sr=sr)
    audio_segment = split_audio(wav, sr, duration)
    #gated_audio = np.where(np.abs(wav) > 0.02, wav, 0)
    #audio_segment = split_audio(wav, sr, duration)
    #stft = librosa.stft(audio_segment[0], n_fft=1024, hop_length=512)
    #magnitude = np.abs(stft)
    #energy = np.sum(magnitude**2, axis=0) 
    #zcr = librosa.feature.zero_crossing_rate(audio_segment[0])

    # Extract pitch (f0) using pYIN
    #f0, voiced_flag, voiced_probs = librosa.pyin(audio_segment[0],   fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    # Replace NaNs in f0 (unvoiced) with 0
    #f0 = np.nan_to_num(f0)


    
    f0 = librosa.yin(audio_segment[0], fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

    # Extract intensity using Root Mean Square (RMS)
    #intensity = librosa.feature.rms(y=audio_segment[0])
    intensity = librosa.feature.rms(y=audio_segment[0], frame_length=1024, hop_length=512)


    # Optionally downsample pitch and intensity (if needed for uniform shape)
    f0_downsampled = downsample_with_max_pooling(f0.reshape(1, -1), (1, 4))
    intensity_downsampled = downsample_with_max_pooling(intensity, (1, 4))_mean
    #energy_downsampled = downsample_with_max_pooling(energy.reshape(1, -1), (1, 4))
    #zcr_downsampled = downsample_with_max_pooling(zcr, (1, 4))

    return f0_downsampled, intensity_downsampled'''


def pitch_and_intensity_from_audio(path, sr, duration):
    wav, sr = librosa.load(path, sr=sr)
    #reduced_noise_audio = nr.reduce_noise(y=wav, sr=sr)
    
    # Split the cleaned audio into segments of the specified duration
    audio_segment = split_audio(wav, sr, duration)
    
    # Split audio into segments of the specified duration
    #audio_segment = split_audio(wav, sr, duration)
    
    # Extract pitch (f0) using YIN
    f0 = librosa.yin(audio_segment[0], fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)

    # Extract intensity using Root Mean Square (RMS)
    intensity = librosa.feature.rms(y=audio_segment[0], frame_length=1024, hop_length=512)

    # Extract Spectral Features
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment[0], sr=sr, n_fft=1024, hop_length=512)

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment[0], sr=sr, roll_percent=0.85, n_fft=1024, hop_length=512)

    # Spectral Flux
    spectral_flux = librosa.onset.onset_strength(y=audio_segment[0], sr=sr, hop_length=512)

    #harmonic = librosa.effects.harmonic(audio_segment[0])
   # percussive = librosa.effects.percussive(audio_segment[0])
    
    # Ensure there is no division by zero by adding a small epsilon
    #epsilon = 1e-6  # Small constant to avoid division by zero
    #hnr = np.mean(harmonic / (percussive + epsilon))

    # Optionally downsample pitch, intensity, spectral centroid, rolloff, and flux (if needed for uniform shape)
    f0_downsampled = downsample_with_max_pooling(f0.reshape(1, -1), (1, 4))
    intensity_downsampled = downsample_with_max_pooling(intensity, (1, 4))
    spectral_centroid_downsampled = downsample_with_max_pooling(spectral_centroid.reshape(1, -1), (1, 4))
    spectral_rolloff_downsampled = downsample_with_max_pooling(spectral_rolloff.reshape(1, -1), (1, 4))
    spectral_flux_downsampled = downsample_with_max_pooling(spectral_flux.reshape(1, -1), (1, 4))
    #hnr_downsampled = downsample_with_max_pooling(hnr.reshape(1, -1), (1, 4))

    return f0_downsampled, intensity_downsampled

def extract_features_parallel(path, sr, duration):
    chroma = chroma_from_audio_file(path, sr, duration)
    tonnetz = tonnetz_from_audio_file(path, sr, duration)
    pitch, intensity = pitch_and_intensity_from_audio(path, sr, duration)
    combined_features = np.concatenate((chroma, tonnetz, pitch, intensity), axis=0)
    return combined_features
    
def process_audio_files_in_parallel(paths, sr, duration):
    results = []

    # Parallel processing of audio files using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = {executor.submit(pitch_and_intensity_from_audio, path, sr, duration): path for path in paths}

        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")

    return results