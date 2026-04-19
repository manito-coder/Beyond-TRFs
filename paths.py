from dependencies import *

# Define download URLs
#AUDIO_URL = 'https://zenodo.org/record/1199011/files/AUDIO.zip'
#DATA_PREPROC_URL = 'https://zenodo.org/record/1199011/files/DATA_preproc.zip'

# Root data directory
DATA_ROOT = Path("~").expanduser() / 'Data' / 'Beyond-TRFs'
ALICE_ROOT = DATA_ROOT / 'Alice'
FUGLSANG_ROOT = DATA_ROOT / 'Fuglsang'

# Fuglsanf matlab directories
#FUGLSANG_DATA_RAW = FUGLSANG_ROOT / 'data_raw'
FUGLSANG_DATA_PREPROC = FUGLSANG_ROOT / 'data_preprocessed'
#FUGLSANG_MAT_FILE_TRF_DIR = FUGLSANG_ROOT / 'mat_file'

# Stimuli directory and list of stimulus names (without file extensions)
ALICE_STIMULUS_DIR = ALICE_ROOT / 'stimuli'
FUGLSANG_STIMULUS_DIR = FUGLSANG_ROOT / 'stimuli'

# Envelopes directory
#ALICE_ENVELOPES_DIR = ALICE_ROOT / "envelopes"
#FUGLSANG_ENVELOPES_DIR = FUGLSANG_ROOT / "envelopes"

# Predictors directory
ALICE_PREDICTOR_DIR = ALICE_ROOT / 'predictors'
FUGLSANG_PREDICTOR_DIR = FUGLSANG_ROOT / 'predictors' / 'concatenated' / 'self_computed'

# Predictors directory
ALICE_PROCESSED_PREDICTOR_DIR = ALICE_ROOT / 'processed-predictors'
#FUGLSANG_PROCESSED_PREDICTOR_DIR = FUGLSANG_ROOT / 'processed-predictors'

# EEG data directory and list of subjects
ALICE_EEG_DIR = ALICE_ROOT / 'eeg'
FUGLSANG_EEG_DIR = FUGLSANG_ROOT / 'eeg'

# Define a target directory for TRF estimates and make sure the directory is created
ALICE_TRF_DIR = ALICE_ROOT / 'TRFs'
FUGLSANG_TRF_DIR = FUGLSANG_ROOT / 'TRFs' / 'self_computed'

# Figures directory
ALICE_FIGURES_DIR = ALICE_ROOT / 'figures'
FUGLSANG_FIGURES_DIR = FUGLSANG_ROOT / 'figures'

# Make sure all directories exist
'''
for directory in [ALICE_ROOT, ALICE_ENVELOPES_DIR, ALICE_PREDICTOR_DIR, ALICE_PROCESSED_PREDICTOR_DIR, ALICE_EEG_DIR, ALICE_TRF_DIR, ALICE_FIGURES_DIR,
                  FUGLSANG_ROOT, FUGLSANG_ENVELOPES_DIR, FUGLSANG_PREDICTOR_DIR, FUGLSANG_PROCESSED_PREDICTOR_DIR, FUGLSANG_EEG_DIR, FUGLSANG_TRF_DIR, FUGLSANG_FIGURES_DIR, FUGLSANG_DATA_RAW, FUGLSANG_DATA_PREPROC]:
    directory.mkdir(parents=True, exist_ok=True)
'''
for directory in [ALICE_ROOT, ALICE_PREDICTOR_DIR, ALICE_TRF_DIR, ALICE_FIGURES_DIR, ALICE_EEG_DIR, ALICE_STIMULUS_DIR, ALICE_PROCESSED_PREDICTOR_DIR, 
                  FUGLSANG_ROOT, FUGLSANG_PREDICTOR_DIR, FUGLSANG_TRF_DIR, FUGLSANG_FIGURES_DIR, FUGLSANG_DATA_PREPROC, FUGLSANG_EEG_DIR, FUGLSANG_STIMULUS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)