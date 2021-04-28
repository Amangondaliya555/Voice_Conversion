import librosa
import numpy as np
import os
import pyworld
import pyworld as pw
import glob
from utility import *
import argparse

FEATURE_DIM = 36
SAMPLE_RATE = 16000
FRAMES = 512
FFTSIZE = 1024
SPEAKERS_NUM = 4  # in our experiment, we use four speakers

EPSILON = 1e-10
MODEL_NAME = 'starganvc_model'


def load_wavs(dataset: str, sr):
    '''
    data dict contains all audios file path
    resdict contains all wav files   
    '''
    data = {}
    with os.scandir(dataset) as it: # just names of files or folders 
        for entry in it:
            if entry.is_dir():  # Checking if it is directory (os feature)
                data[entry.name] = [] # folder name. (here speaker name)
                # print(entry.name, entry.path)
                with os.scandir(entry.path) as it_f: # (path of the folders)
                    for onefile in it_f: # iterating folders to acces files
                        if onefile.is_file(): # Checking if it is file (os feature)
                            # print(onefile.path)
                            data[entry.name].append(onefile.path) # making dictionary according to folder and files
    print(f'loaded keys: {data.keys()}')
    # data like {TM1:[xx,xx,xxx,xxx]}
    resdict = {}

    cnt = 0 # counting total wav files (all speakers)
    for key, value in data.items(): # iterating dictionary (key is speaker name (folder) and value is wav files of that speaker)
        resdict[key] = {} # same key as data dict. for resdict (key = speaker name(folder))

        for one_file in value: # iterting through wav files of particular speaker

            filename = os.path.normpath(one_file).split(os.sep)[-1].split('.')[0]  # like 10006, A//B, A/B/, A/./B and A/foo/../B all will be normalized to A/B # os.sep return path separator accroding to the os. (like '\' for windows)
            newkey = f'{filename}' # format string
            wav, _ = librosa.load(one_file, sr=sr, mono=True, dtype=np.float64)
            """
            - Mono or monophonic audio describes a mix in which all sounds are mixed together into a single channel. One character for an 8-bit mono signal.
            - librosa.load() returns sampling rate and a 2D array
            - The first axis: represents the recorded samples of amplitudes (change of air pressure) in the audio. The second axis: represents the number of channels in the audio.
            - wav is 2D array here
            """

            resdict[key][newkey] = wav
            # resdict[key].append(temp_dict) #like TM1:{100062:[xxxxx], .... }
            print('.', end='')
            cnt += 1

    print(f'\nTotal {cnt} aduio files!')
    return resdict


def wav_to_mcep_file(dataset: str, sr=16000, ispad: bool = False, processed_filepath: str = './data/processed'):
    '''convert wavs to mcep feature using image repr'''
    # if no processed_filepath, create it ,or delete all npz files
    if not os.path.exists(processed_filepath):
        os.makedirs(processed_filepath)
    else:
        filelist = glob.glob(os.path.join(processed_filepath, "*.npy"))
        for f in filelist:
            os.remove(f)

    allwavs_cnt = len(glob.glob(f'{dataset}/*/*.wav'))
    # allwavs_cnt = allwavs_cnt//4*3 * 12+200 #about this number not precise
    print(f'Total {allwavs_cnt} audio files!')

    d = load_wavs(dataset, sr) # dict. like TM1:{100062:[xxxxx], .... } 
    cnt = 1  #

    for one_speaker in d.keys(): # iterating through speakers(TM1, TM2...)
        for audio_name, audio_wav in d[one_speaker].items(): #audio names like 100062 (key) and audio waves is 2D array
            # cal source audio feature
            audio_mcep_dict = cal_mcep(
                audio_wav, fs=sr, ispad=ispad, frame_period=0.005, dim=FEATURE_DIM) # receive dict. of features
            newname = f'{one_speaker}-{audio_name}' # like : TM1-100062

            # save the dict as npz
            file_path_z = f'{processed_filepath}/{newname}' # saved at ./data/processed/
            print(f'save file: {file_path_z}')
            np.savez(file_path_z, audio_mcep_dict)

            # save every  36*FRAMES blocks
            print(f'audio mcep shape {audio_mcep_dict["coded_sp"].shape}')

            # TODO step may be FRAMES//2
            for start_idx in range(0, audio_mcep_dict["coded_sp"].shape[1] - FRAMES + 1, FRAMES):
                one_audio_seg = audio_mcep_dict["coded_sp"][:,
                                                            start_idx:start_idx + FRAMES]

                if one_audio_seg.shape[1] == FRAMES:

                    temp_name = f'{newname}_{start_idx}'
                    filePath = f'{processed_filepath}/{temp_name}'

                    print(f'[{cnt}:{allwavs_cnt}]svaing file: {filePath}.npy')
                    np.save(filePath, one_audio_seg)
            cnt += 1


def cal_mcep(wav_ori, fs=SAMPLE_RATE, ispad=False, frame_period=0.005, dim=FEATURE_DIM, fft_size=FFTSIZE):
    '''cal mcep given wav singnal
        the frame_period used only for pad_wav_to_get_fixed_frames
    '''
    if ispad:
        wav, pad_length = pad_wav_to_get_fixed_frames(
            wav_ori, frames=FRAMES, frame_period=frame_period, sr=fs)
    else:
        wav = wav_ori

    """
    The FFT size defines the number of bins used for dividing the window into equal strips, or bins. Hence, a bin is a spectrum sample, and defines the frequency resolution of the window.

    By default :

    N (Bins) = FFT Size/2

    FR = Fmax/N(Bins)

    For a 44100 sampling rate, we have a 22050 Hz band. With a 1024 FFT size, we divide this band into 512 bins.

    FR = 22050/1024 ≃ 21,53 Hz.
    """

    """
    An audio frame, or sample, contains amplitude (loudness) information at that particular point in time.
    To produce sound, tens of thousands of frames are played in sequence to produce frequencies.
    """
    # Harvest F0 extraction algorithm.
    f0, timeaxis = pyworld.harvest(wav, fs) # Harvest estimates F0 trajectory given a monoral input signal. returns time_axis : Temporal positions and f :F0 contour.

    """
    The fundamental frequency or F0 is the frequency at which vocal chords vibrate in voiced sounds.
    This frequency can be identified in the sound produced, which presents quasi-periodicity,
    the pitch period being the fundamental period of the signal (the inverse of the fundamental frequency).
    """
    
    # CheapTrick harmonic spectral envelope estimation algorithm.
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs, fft_size=fft_size) # CheapTrick calculates the spectrogram that consists of spectral envelopes estimated by CheapTrick. return spectrogram.

    """
    CheapTrick consists of power spectrum estimation with the F0-adaptive Hanning window, the smoothing of the power spectrum, and spectral recovery in the quefrency domain.
    The algorithm can obtain an accurate and temporally stable spectral envelope by objective evaluations.
    """
    
    # D4C aperiodicity estimation algorithm.
    ap = pyworld.d4c(wav, f0, timeaxis, fs, fft_size=fft_size) # D4C calculates the aperiodicity estimated by D4C.
    """
    An algorithm is proposed for estimating the band aperiodicity of speech signals,
    where “aperiodicity” is defined as the power ratio between the speech signal and the aperiodic component of the signal.
    """
    
    # feature reduction nxdim
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim) # CodeSpectralEnvelope codes the spectral envelope.
    # log
    coded_sp = coded_sp.T  # dim x n

    res = {
        'f0': f0,  # n
        'ap': ap,  # n*fftsize//2+1
        'sp': sp,  # n*fftsize//2+1
        'coded_sp': coded_sp,  # dim * n
    }
    return res


def pad_wav_to_get_fixed_frames(x: np.ndarray, frames: int = 128, frame_period: float = 0.005, sr: int = 16000):
    # one frame's points
    frame_length = frame_period * sr
    # frames points
    frames_points = frames * frame_length

    wav_len = len(x)

    # pad amount
    pieces = wav_len // frames_points

    need_pad = 0
    if wav_len % frames_points != 0:
        # can't devide need pad
        need_pad = int((pieces + 1) * frames_points - wav_len)

    afterpad_len = wav_len + need_pad
    # print(f'need pad: {need_pad}, after pad: {afterpad_len}')
    # padding process
    tempx = x.tolist()

    if need_pad <= len(x):
        tempx.extend(x[:need_pad])
    else:
        temp1, temp2 = need_pad // len(x), need_pad / len(x)
        tempx = tempx * (temp1 + 1)
        samll_pad_len = int(np.ceil((temp2 - temp1) * len(x)))
        tempx.extend(x[:samll_pad_len])

        diff = 0
        if afterpad_len != len(tempx):
            diff = afterpad_len - len(tempx)
        if diff > 0:
            tempx.extend(tempx[:diff])
        elif diff < 0:
            tempx = tempx[:diff]

    # print(f'padding length: {len(x)}-->length: {len(tempx)}')
    # remove last point for calculate convience:the frame length are 128*(some integer).
    tempx = tempx[:-1]

    return np.asarray(tempx, dtype=np.float), need_pad


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert the wav waveform to mel-cepstral coefficients(MCCs)\
    and calculate the speech statistical characteristics')

    input_dir = './data/fourspeakers'
    output_dir = './data/processed'
    ispad = True

    parser.add_argument('--input_dir', type=str,
                        help='the direcotry contains data need to be processed', default=input_dir)
    parser.add_argument('--output_dir', type=str,
                        help='the directory stores the processed data', default=output_dir)
    parser.add_argument(
        '--ispad', type=bool, help='whether to pad the wavs  to get fixed length MCEP', default=ispad)

    argv = parser.parse_args()
    input_dir = argv.input_dir
    output_dir = argv.output_dir
    ispad = argv.ispad

    wav_to_mcep_file(input_dir, SAMPLE_RATE, ispad=ispad,
                     processed_filepath=output_dir)

    # input_dir is train dataset. we need to calculate and save the speech\
    # statistical characteristics for each speaker.
    generator = GenerateStatics(output_dir)
    generator.generate_stats()
