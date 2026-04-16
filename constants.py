from dependencies import *

# =============================================================================
# ENUMS
# =============================================================================

class PREDICTOR_TYPE(Enum):
    ENVELOPE = "envelope"
    ENVELOPE_ONSET = "envelope_onset"

class ATTENTION_TYPE(Enum):
    ATTENDED = "attended"
    IGNORED  = "ignored"
    SINGLE = ""

class MODEL_TYPE(Enum):
    FORWARD  = "forward"
    BACKWARD = "backward"

class DATASET_TYPE(Enum):
    FUGLSANG = "fuglsang"
    ALICE = "alice"

# =============================================================================
# DOWNLOAD URLs
# =============================================================================

AUDIO_URL        = 'https://zenodo.org/record/1199011/files/AUDIO.zip'
DATA_PREPROC_URL = 'https://zenodo.org/record/1199011/files/DATA_preproc.zip'
