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
    trf_data  = {get_trf_model_name(dataset, p, a, m): [] for p, a, m in checks}
    skipped   = []
    loaded    = []

    for subject in subjects:
        missing = []
        for p, a, m in checks:
            if dataset == DATASET_TYPE.FUGLSANG:
                if not (trf_dir / subject / f"{subject}_{get_trf_model_name(dataset, p, a, m)}_trf.pickle").exists():
                    missing.append(get_trf_model_name(dataset, p, a, m))
            elif dataset == DATASET_TYPE.ALICE:
                if not (trf_dir / subject / f"{subject} 64hz-{get_trf_model_name(dataset, p, a, m)}.pickle").exists():
                    missing.append(get_trf_model_name(dataset, p, a, m))
        '''
        missing = [
            get_trf_model_name(dataset, p, a, m)
            for p, a, m in checks
            if not (trf_dir / subject / f"{subject}_{get_trf_model_name(dataset, p, a, m)}_trf.pickle").exists()
        ]
        '''

        if missing:
            print(f"  ✗ {subject}: skipping — missing {len(missing)} TRF(s):")
            for name in missing:
                print(f"      - {name}")
            skipped.append(subject)
            continue

        for p, a, m in checks:
            name = get_trf_model_name(dataset, p, a, m)
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


def get_predictor_name(predictors, padded=False) -> str:
    """
    Format:
        <predictor1+predictor2+...>[ _padded ]

    Example:
        envelope+envelope_onset_padded
    """
    if isinstance(predictors, PREDICTOR_TYPE):
        predictors = [predictors]

    predictors = sorted(predictors, key=lambda p: p.value)
    name = "+".join(p.value for p in predictors)

    if padded:
        name += "_padded"

    return name

def get_attentional_predictor_name(predictors, attention: ATTENTION_TYPE, padded=False) -> str:
    """
    Format:
        <attention_type>_<predictor_combination>

    Example:
        attended_envelope+envelope_onset_padded
    """
    return f"{attention.value}_{get_predictor_name(predictors, padded)}"

    

def get_trf_model_name(
    dataset: DATASET_TYPE,
    predictors: PREDICTOR_TYPE | list[PREDICTOR_TYPE],
    attention: ATTENTION_TYPE,
    model: MODEL_TYPE,
    generalised: GENERALISATION_TYPE = GENERALISATION_TYPE.INDIVIDUAL,
    padded: bool = False
):
    """
    Generate standardized TRF model names.

    Format:
        [<generalisation_type>]_<model_type>_<trf_type>_<predictor1+predictor2>[ _padded ]

    Example:
        backward_attended_envelope+envelope_onset_padded
    """
    if isinstance(predictors, PREDICTOR_TYPE):
        predictors = [predictors]

    predictors = sorted(predictors, key=lambda p: p.value)
    predictor_names = "+".join(
        map_predictor_name(p, dataset) for p in predictors
    )

    if dataset == DATASET_TYPE.FUGLSANG:
        name = f"{model.value}_{attention.value}_{predictor_names}"
        if padded:
            name += "_padded"

    elif dataset == DATASET_TYPE.ALICE:
        parts = []
        if model == MODEL_TYPE.BACKWARD:
            parts.append("decoder")
        parts.append(predictor_names)
        name = "-".join(parts)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Prepend generalisation prefix if not individual
    if generalised != GENERALISATION_TYPE.INDIVIDUAL:
        name = f"{generalised.value}_{name}"

    return name



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
# AAD FUNCTIONS

def aad_single_classifier(eeg, true_att, true_ign, trf):
    """
    One generic TRF: reconstruct once, correlate against both stimuli.
    Returns True if reconstruction correlates more with attended stimulus.
    """
    pred = eelbrain.convolve(trf.h_scaled, eeg).x

    att = true_att.x if hasattr(true_att, 'x') else np.asarray(true_att)
    ign = true_ign.x if hasattr(true_ign, 'x') else np.asarray(true_ign)

    # convolve can produce pred that is 1 sample shorter than the envelopes
    n    = min(pred.shape[-1], att.shape[-1], ign.shape[-1])
    pred = pred[..., :n]
    att  = att[...,  :n]
    ign  = ign[...,  :n]

    r_att = np.abs(np.corrcoef(pred, att)[0, 1])
    r_ign = np.abs(np.corrcoef(pred, ign)[0, 1])
    return r_att > r_ign, r_att, r_ign


def aad_double_classifier(eeg, true_att, true_ign, att_trf, ign_trf):
    """
    Two condition-specific TRFs: reconstruct attended and ignored separately,
    correlate each with its respective true stimulus.
    Returns True if attended reconstruction wins.
    """
    pred_att = eelbrain.convolve(att_trf.h_scaled, eeg).x
    pred_ign = eelbrain.convolve(ign_trf.h_scaled, eeg).x

    att = true_att.x if hasattr(true_att, 'x') else np.asarray(true_att)
    ign = true_ign.x if hasattr(true_ign, 'x') else np.asarray(true_ign)

    # each convolve call can drift independently, so trim each pair separately
    n_att = min(pred_att.shape[-1], att.shape[-1])
    n_ign = min(pred_ign.shape[-1], ign.shape[-1])

    r_att = np.abs(np.corrcoef(pred_att[..., :n_att], att[..., :n_att])[0, 1])
    r_ign = np.abs(np.corrcoef(pred_ign[..., :n_ign], ign[..., :n_ign])[0, 1])

    return r_att > r_ign, r_att, r_ign


def aad_classifier(predictors, subjects, 
                   generalised=GENERALISATION_TYPE.AVERAGE, 
                   cv=CROSS_VALIDATION_TYPE.HOLD_OUT, aad_type = AAD_APPROACH.SINGLE):

    att_predictor_name = get_attentional_predictor_name(predictors, ATTENTION_TYPE.ATTENDED)
    ign_predictor_name = get_attentional_predictor_name(predictors, ATTENTION_TYPE.IGNORED)

    att_trf_name = get_trf_model_name(DATASET_TYPE.FUGLSANG, predictors, ATTENTION_TYPE.ATTENDED, MODEL_TYPE.BACKWARD, generalised=generalised)

    if (aad_type == AAD_APPROACH.DOUBLE):
        ign_trf_name = get_trf_model_name(DATASET_TYPE.FUGLSANG, predictors, ATTENTION_TYPE.IGNORED,  MODEL_TYPE.BACKWARD, generalised=generalised)

    # For hold-out, one TRF for all subjects — load once
    if cv == CROSS_VALIDATION_TYPE.HOLD_OUT:
        trf_att = eelbrain.load.unpickle(FUGLSANG_GENERAL_TRF_DIR / f'hold_out_{att_trf_name}.pickle')
        if (aad_type == AAD_APPROACH.DOUBLE):
            trf_ign = eelbrain.load.unpickle(FUGLSANG_GENERAL_TRF_DIR / f'hold_out_{ign_trf_name}.pickle')

    decisions = {}
    r_atts    = {}
    r_igns    = {}

    for subject in subjects:

        # For LOO, load the TRF that excluded this subject
        if cv == CROSS_VALIDATION_TYPE.LOO:
            trf_att = eelbrain.load.unpickle(FUGLSANG_GENERAL_TRF_DIR / f'loocv_{subject}_{att_trf_name}.pickle')
            if (aad_type == AAD_APPROACH.DOUBLE):
                trf_ign = eelbrain.load.unpickle(FUGLSANG_GENERAL_TRF_DIR / f'loocv_{subject}_{ign_trf_name}.pickle')

        eeg = eelbrain.load.unpickle(
            FUGLSANG_EEG_DIR / subject / f'{subject}_eeg.pickle'
        )
        true_att = eelbrain.load.unpickle(
            FUGLSANG_PREDICTOR_DIR / subject / f'{att_predictor_name}_concat.pickle'
        ).x
        true_ign = eelbrain.load.unpickle(
            FUGLSANG_PREDICTOR_DIR / subject / f'{ign_predictor_name}_concat.pickle'
        ).x

        if (aad_type == AAD_APPROACH.DOUBLE):
            decision, r_att, r_ign = aad_double_classifier(eeg, true_att, true_ign, trf_att, trf_ign)
        else:
            decision, r_att, r_ign = aad_single_classifier(eeg, true_att, true_ign, trf_att)
        decisions[subject] = decision
        r_atts[subject]    = r_att
        r_igns[subject]    = r_ign
        print(f"{subject}: r_att={r_att:.3f}, r_ign={r_ign:.3f}")

    acc = sum(decisions.values()) / len(subjects)
    print(f"\n✅ Classification rate ({predictors}): {acc:.2%}")
    print('\n' + '─' * 60 + '\n')

    return acc, decisions, r_atts, r_igns






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