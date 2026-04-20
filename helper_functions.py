from dependencies import *
from constants import *
from paths import *

# ————————————————————————————————————————————————————————————————————————————————————————————
# GLOBAL FUNCTIONS

def load_trfs(dataset, subjects, checks, trf_dir):
    """
    Load TRFs for all subjects and checks.

    Returns:
        trf_data:   dict keyed by model_name -> list of TRFs (one per subject)
        n_subjects: number of subjects successfully loaded
    """
    subjects  = subjects
    trf_data  = {get_trf_model_name(dataset, p, a, m, pad): [] for p, a, m, pad in checks}
    skipped   = []
    loaded    = []

    for subject in subjects:
        missing = []
        for p, a, m, pad in checks:
            if dataset == DATASET_TYPE.FUGLSANG:
                if not (trf_dir / subject / f"{subject}_{get_trf_model_name(dataset, p, a, m, pad)}_trf.pickle").exists():
                    missing.append(get_trf_model_name(dataset, p, a, m, pad))
            elif dataset == DATASET_TYPE.ALICE:
                if not (trf_dir / subject / f"{subject} 64hz-{get_trf_model_name(dataset, p, a, m, pad)}.pickle").exists():
                    missing.append(get_trf_model_name(dataset, p, a, m, pad))
        '''
        missing = [
            get_trf_model_name(dataset, p, a, m, pad)
            for p, a, m, pad in checks
            if not (trf_dir / subject / f"{subject}_{get_trf_model_name(dataset, p, a, m, pad)}_trf.pickle").exists()
        ]
        '''

        if missing:
            print(f"  ✗ {subject}: skipping — missing {len(missing)} TRF(s):")
            for name in missing:
                print(f"      - {name}")
            skipped.append(subject)
            continue

        for p, a, m, pad in checks:
            name = get_trf_model_name(dataset, p, a, m, pad)
            if dataset == DATASET_TYPE.FUGLSANG:
                path = trf_dir / subject / f"{subject}_{name}_trf.pickle"
            elif dataset == DATASET_TYPE.ALICE:
                path = trf_dir / subject / f"{subject} 64hz-{name}.pickle"
            trf_data[name].append(eelbrain.load.unpickle(path))

        loaded.append(subject)
        print(f"  ✓ {subject}")

    print(f"\nLoaded: {len(loaded)} subjects | Skipped: {len(skipped)} subjects")
    if skipped:
        print(f"  Skipped: {skipped}")

    n_subjects = len(loaded)
    return trf_data, n_subjects



def get_trf_model_name(dataset: DATASET_TYPE, predictors: PREDICTOR_TYPE|list[PREDICTOR_TYPE], attention: ATTENTION_TYPE, model: MODEL_TYPE, padded: bool=False):
    """
    Generate standardized TRF model names.

    Format:
        <model_type>_<trf_type>_<predictor1+predictor2>[ _padded ]

    Example:
        backward_attended_envelope+envelope_onset_padded
    """

    # ————————————————————————————————————————————————————
    # GENERALISE PREDICTORS
    if isinstance(predictors, PREDICTOR_TYPE):
        predictors = [predictors]

    predictors = sorted(predictors, key=lambda p: p.value)
    predictor_names = "+".join(
        map_predictor_name(p, dataset) for p in predictors
    )

    # ————————————————————————————————————————————————————
    # BLOCK FOR BUILDING TRF NAMES IN THE FUGLSANG DATASET
    if dataset == DATASET_TYPE.FUGLSANG:
        name = f"{model.value}_{attention.value}_{predictor_names}"
        if padded:
            name += "_padded"

        return name
    
    # ————————————————————————————————————————————————————
    # BLOCK FOR BUILDING TRF NAMES IN THE ALICE DATASET
    elif dataset == DATASET_TYPE.ALICE:
        parts = []
        # Only include "decoder" for backward
        if model == MODEL_TYPE.BACKWARD:
            parts.append("decoder")
        # Alice ignores attention
        parts.append(predictor_names)
        name = "-".join(parts)

        return name

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    


def map_predictor_name(predictor: PREDICTOR_TYPE, dataset: DATASET_TYPE):
    if dataset == DATASET_TYPE.FUGLSANG:
        return predictor.value

    elif dataset == DATASET_TYPE.ALICE:
        mapping = {
            PREDICTOR_TYPE.ENVELOPE: "envelope_log",
            PREDICTOR_TYPE.ENVELOPE_ONSET: "envelope_onset",
        }
        return mapping[predictor]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

# ————————————————————————————————————————————————————————————————————————————————————————————
# FUGLSANG FUNCTIONS

# Utility function to get subject list 
def fuglsang_get_subjects():
    subjects = [path.stem.split("_")[0] for path in FUGLSANG_DATA_PREPROC.glob("*.mat")]
    subjects = sorted(subjects, key=lambda x: int(re.search(r'S(\d+)', x).group(1)))
    return subjects

# ————————————————————————————————————————————————————————————————————————————————————————————
# ALICE FUNCTIONS

# Utility function to get subject list
def alice_get_subjects():
    subjects = [path.name for path in ALICE_EEG_DIR.iterdir() if path.is_dir()]
    subjects = sorted(subjects, key=lambda x: int(re.search(r'S(\d+)', x).group(1)))
    return subjects

def alice_get_durations(envelope, STIMULI):
    durations = [gt.time.tmax for stimulus, gt in zip(STIMULI, envelope)]
    return durations


# ————————————————————————————————————————————————————————————————————————————————————————————
# PLOTTING FUNCTIONS

def set_plot_style():
    FONT      = 'Arial'
    FONT_SIZE = 8
    RC = {
        'figure.dpi':          100,
        'savefig.dpi':         300,
        'savefig.transparent': True,
        'font.family':         'sans-serif',
        'font.sans-serif':     FONT,
        'font.size':           FONT_SIZE,
        'figure.labelsize':    FONT_SIZE,
        'figure.titlesize':    FONT_SIZE,
        'axes.labelsize':      FONT_SIZE,
        'axes.titlesize':      FONT_SIZE,
        'xtick.labelsize':     FONT_SIZE,
        'ytick.labelsize':     FONT_SIZE,
        'legend.fontsize':     FONT_SIZE,
    }
    plt.rcParams.update(RC)


# Utility function to get significance marker based on p-value --------------------------------------------------
def sig_marker(p):
        if p < 0.001:  return '***'
        elif p < 0.01: return '**'
        elif p < 0.05: return '*'
        else:          return 'n.s.'

# Utility function to add significance lines to plots --------------------------------------------------
def add_sig_line(ax, x1, x2, y, text, color='k'):
    ax.plot([x1, x1, x2, x2], [y, y + 0.0005, y + 0.0005, y], color=color, linewidth=1)
    ax.text((x1 + x2) / 2, y + 0.0005, text, ha='center', va='bottom', fontsize=10)

def lighten_color(color, amount=0.7):
    """
    Lighten a matplotlib color.

    Parameters
    ----------
    color : str or tuple
        Any matplotlib-compatible color.
    amount : float
        0 → original color
        1 → white

    Returns
    -------
    tuple
        Lightened RGB color
    """
    c = colors.to_rgb(color)
    return tuple(1 - (1 - x) * (1 - amount) for x in c)