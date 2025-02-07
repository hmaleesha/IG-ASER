import glob
import os
import re
import shutil
import os
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import tqdm
import librosa

from audio_utils import mfcc_from_audio_file, chroma_from_audio_file, tonnetz_from_audio_file, tonnetzchroma__from_audio, pitch_and_intensity_from_audio, spectrogram_from_audio_file

k_folds_exclusions = {
    1: ['01_M', '02_F'],
    2: ['02_M', '03_F'],
    3: ['03_M', '04_F'],
    4: ['04_M', '05_F'],
    5: ['05_M', '01_F']
}



EMOTIONS = ['hap', 'sad', 'ang', 'neu']


def get_impro_wav_folders(iemocap_path):
    wav_folders = glob.glob(iemocap_path + '/Session*/sentences/wav/*')
    #print(wav_folders)
    impro_wav_folders = [f for f in wav_folders if "impro" in f]
    processing_ids = []
    processing_wav_folders = []

    for f in impro_wav_folders:
        folder_id = f.split("/")[-1]
        result = re.search("(Ses\d{2})([F|M])_(impro\d{2})", folder_id)
        #result = re.search("(Ses\d{2})M_(impro\d{2})", folder_id)
        if result is not None:
            i = f"{result.group(1)}_{result.group(3)}"
            if i not in processing_ids:
                processing_ids.append(i)
                processing_wav_folders.append(f)

    return processing_wav_folders

'''def get_impro_wav_folders(iemocap_path):
    wav_folders = glob.glob(iemocap_path + '/Session*/sentences/wav/*')
    valid_folders = []
    for folder in wav_folders:
        if ("impro" in folder or "script" in folder) and glob.glob(f"{folder}/*.wav"):
            # Only add folders that contain .wav files
            valid_folders.append(folder)
    return valid_folders'''


def get_wav_files(wav_folders):
    wav_files = []
    for f in wav_folders:
        wav = glob.glob(f'{f}/*')
        wav_files.extend(wav)

    return wav_files


def get_speaker(fname):
    sections = fname.split('/')
    sentence_id = sections[-1].split(".")[0]
    result = re.search("Ses(\d{2}).*([F|M])\d{3}", sentence_id)
    #result = re.search("(Ses\d{2})M_(impro\d{2})", sentence_id)
    return f"{result.group(1)}_{result.group(2)}"


def get_audio_files_per_speaker(wav_files):
    audio_files = {}
    for f in wav_files:
        speaker = get_speaker(f)
        try:
            audio_files[speaker]
        except KeyError:
            audio_files[speaker] = []

        audio_files[speaker].append(f)

    return audio_files


def get_training_files_per_fold(fold, audio_files_per_speaker):
    files = []
    for s in audio_files_per_speaker:
        if s not in k_folds_exclusions[fold]:
            files.extend(audio_files_per_speaker[s])

    return files


def get_val_files_per_fold(fold, audio_files_per_speaker):
    files = []
    for s in audio_files_per_speaker:
        if s in k_folds_exclusions[fold]:
            files.extend(audio_files_per_speaker[s])

    return files


def get_class(fname):
    sections = fname.split('\\')
    session_id = sections[-5]
    dialog_id = sections[-2]
    sentence_id = sections[-1].split(".")[0]

    dataset_base_path = str.join("/", fname.split("\\")[0:-5])

    emo_evaluation_file = dataset_base_path + '/' + session_id + '/dialog/EmoEvaluation/' + dialog_id + '.txt'
    with open(emo_evaluation_file, 'r') as f:
        targets = [line for line in f if sentence_id in line]
        emo = targets[0].split('\t')[2]
        return emo


def copy_file(src_file_path, destination_base_path):
    c = get_class(src_file_path)
    d = destination_base_path + "/" + c + "/" + str.join("/", src_file_path.split("\\")[-1:])
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy(src_file_path, d)


'''def save_mfcc(src_file_path, destination_base_path, sample_rate, utterance_duration):
    #spectrogram = mfcc_from_audio_file(src_file_path, sample_rate, utterance_duration)
    spectrogram = tonnetz_from_audio_file(src_file_path, sample_rate, utterance_duration)

    c = get_class(src_file_path)
    d = destination_base_path + "/" + c + "/" + str.join("/", src_file_path.split("\\")[-1:])
    os.makedirs(os.path.dirname(d), exist_ok=True)
    with open(f"{d}.npy", "wb") as f:
        np.save(f, spectrogram)'''

'''def save_mfcc(src_file_path, destination_base_path, sample_rate, utterance_duration):
    # Extract MFCC features
    mfcc = mfcc_from_audio_file(src_file_path, sample_rate, utterance_duration)

    # Compute Delta (first derivative) and Delta-Delta (second derivative) features
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    #chroma, tonnetz = tonnetzchroma__from_audio(src_file_path, sample_rate, utterance_duration)
    # Concatenate MFCC, Delta, and Delta-Delta features along the first axis
    #combined_features = np.concatenate((mfcc, chroma, tonnetz), axis=0)
    combined_features = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)

    # Get the emotion class of the audio
    c = get_class(src_file_path)
    
    # Set destination path and create directories if they don't exist
    d = destination_base_path + "/" + c + "/" + str.join("/", src_file_path.split("\\")[-1:])
    os.makedirs(os.path.dirname(d), exist_ok=True)
    
    # Save the concatenated features as a .npy file
    with open(f"{d}.npy", "wb") as f:
        np.save(f, combined_features)'''

def save_mfcc(src_file_path, destination_base_path, sample_rate, utterance_duration):
    # Extract MFCC features
    mfcc = mfcc_from_audio_file(src_file_path, sample_rate, utterance_duration)

    # Compute Delta (first derivative) and Delta-Delta (second derivative) features
    #delta_mfcc = librosa.feature.delta(mfcc)
    #delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    #chroma, tonnetz = tonnetzchroma__from_audio(src_file_path, sample_rate, utterance_duration)
    # Extract Pitch (Fundamental Frequency)
    #f0, intensity = pitch_and_intensity_from_audio(src_file_path, sample_rate, utterance_duration)
    # Combine MFCC, Delta, Delta-Delta, Pitch, and Intensity features
    combined_features = np.concatenate((mfcc), axis=0)

    # Get the emotion class of the audio
    c = get_class(src_file_path)

    # Set destination path and create directories if they don't exist
    d = destination_base_path + "/" + c + "/" + str.join("/", src_file_path.split("\\")[-1:])
    os.makedirs(os.path.dirname(d), exist_ok=True)
   

    # Save the concatenated features as a .npy file
    with open(f"{d}.npy", "wb") as f:
        np.save(f, mfcc)

'''def save_mfcc(src_file_path, destination_base_path, sample_rate, utterance_duration):
   
    mfcc = spectrogram_from_audio_file(src_file_path, sample_rate, utterance_duration)

    # Get the emotion class of the audio
    c = get_class(src_file_path)

    # Set destination path and create directories if they don't exist
    d = destination_base_path + "/" + c + "/" + str.join("/", src_file_path.split("\\")[-1:])
    os.makedirs(os.path.dirname(d), exist_ok=True)
   

    # Save the concatenated features as a .npy file
    with open(f"{d}.npy", "wb") as f:
        np.save(f, mfcc)'''



def main(iemocap_path, destination):
    wav_folders = get_impro_wav_folders(iemocap_path)
    wav_files = get_wav_files(wav_folders)
    #print(wav_folders)
    files_per_speaker = get_audio_files_per_speaker(wav_files)
    for fold in k_folds_exclusions:
        destination_base_path = f"{destination}/{fold}"
        print(f"Processing fold {fold}")
        training_files = get_training_files_per_fold(fold, files_per_speaker)
        for f in tqdm.tqdm(training_files):
            c = get_class(f)
            if c in EMOTIONS:
                copy_file(f, destination_base_path + "/raw/train")
                save_mfcc(f, destination_base_path + "/iemocapmfcctests/train", 32750, 5)

        val_files = get_val_files_per_fold(fold, files_per_speaker)
        for f in tqdm.tqdm(val_files):
            c = get_class(f)
            if c in EMOTIONS:
                copy_file(f, destination_base_path + "/raw/val")
                save_mfcc(f, destination_base_path + "/iemocapmfcctests/val", 32750, 5)


if __name__ == "__main__":
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
  
    IEMOCAP_PATH = os.path.join(grandparent_dir,"IEMOCAP_full")
    #DESTINATION_PATH = "iemocap_processed_mfcc_k_fold"
    DESTINATION_PATH = "processed_iemocapmfcctests_k_fold"
    main(IEMOCAP_PATH, DESTINATION_PATH)
