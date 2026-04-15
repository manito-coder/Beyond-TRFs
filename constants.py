from dependencies import *

# =============================================================================
# ENUMS
# =============================================================================

class PREDICTOR_TYPE(Enum):
    ENVELOPE                 = "envelope"
    ENVELOPE_ONSET           = "envelope_onset"

class ATTENTION_TYPE(Enum):
    ATTENDED = "attended"
    IGNORED  = "ignored"

class MODEL_TYPE(Enum):
    FORWARD  = "forward"
    BACKWARD = "backward"

# =============================================================================
# DOWNLOAD URLs
# =============================================================================

AUDIO_URL        = 'https://zenodo.org/record/1199011/files/AUDIO.zip'
DATA_PREPROC_URL = 'https://zenodo.org/record/1199011/files/DATA_preproc.zip'
