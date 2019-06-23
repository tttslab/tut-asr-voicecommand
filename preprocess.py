import librosa
import numpy as np
import os
import re
import hashlib
import logging
import argparse
import pickle
import soundfile as sf

logging.basicConfig(level=logging.INFO)

# Parameter setting
# example of tuning by argparser: python train.py --MFCC_DIM=40
parser = argparse.ArgumentParser()
parser.add_argument('--MFCC_DIM',      type=int, default=13)                           # input mfcc dim
parser.add_argument('--WAVE_DIR',      type=str)   # Top directory of speech_commands
parser.add_argument('--TXT_DIR',      type=str)   # txt(listfile) output
parser.add_argument('--MFCC_DIR',      type=str)   # mfcc output
args = parser.parse_args()

wavedir       = args.WAVE_DIR + "/"
txtdir        = args.TXT_DIR + "/"
mfccdir       = args.MFCC_DIR + "/"
datalist      = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
SAMPLING_RATE = 16000
MFCC_DIM      = args.MFCC_DIM

# This function is the same as the code in README.md of speech_commands
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name).encode('utf-8')
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


logging.info('Start making mfcc')
os.makedirs(txtdir, exist_ok=True)
os.makedirs(mfccdir, exist_ok=True)
training_list   = open(txtdir+'train.txt',   'a')
testing_list    = open(txtdir+'eval.txt',    'a')
validation_list = open(txtdir+'valid.txt',   'a')
result = {}

if os.path.exists(mfccdir+'mfcc.pkl'):
    logging.info('mfcc data is already prepared')
    exit()

for command in datalist:
    logging.info('Processing `'+command+'`... ')
    for wavfile in os.listdir(wavedir+command):
        audio, sr = sf.read(wavedir+command+'/'+wavfile)
        mfcc      = librosa.feature.mfcc(audio, sr=SAMPLING_RATE, n_mfcc=MFCC_DIM, n_fft=400, hop_length=160)  # extract mfcc feature 
        mfcc      = np.asarray(mfcc, dtype=np.float32)  # change format to np.float32
        filename = mfccdir+command+'/'+wavfile[:-4]+'.npy'
        result[filename] = mfcc.T
        partition = which_set(wavfile, 10, 10)  # divide to "training", "validation", "testing" 3 parts
        if partition == 'training':
            training_list.write(command+'/'+wavfile[:-4]+'.npy\n')
        if partition == 'testing':
            testing_list.write(command+'/'+wavfile[:-4]+'.npy\n')
        if partition == 'validation':
            validation_list.write(command+'/'+wavfile[:-4]+'.npy\n')
with open(mfccdir+'mfcc.pkl', 'wb') as f:
    pickle.dump(result, f)
training_list.close()
testing_list.close()
validation_list.close()
logging.info('Done')
