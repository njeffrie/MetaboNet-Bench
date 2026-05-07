# Pre-import lightgbm before anything that drags in torch so the Ridge/LightGBM
# Hub paths work on macOS (libomp clashes with torch's OpenMP otherwise; see
# models/lightgbm.py). The import is optional — if lightgbm isn't installed,
# the LightGBM Hub model just won't be usable.
try:
    import lightgbm  # noqa: F401
except ImportError:
    pass

from models.gluformer import Gluformer
from models.zoh import ZeroOrderHold
from models.linear import LinearExtrapolation
from models.lstm import LSTM
from models.UniTS import UniTS
from models.gluforecast import GluForecast

ABLATION_CKPT_DIR = 'checkpoints'

HF_HUB_RIDGE = 'anonymous-4FAD/Ridge'
HF_HUB_LIGHTGBM = 'anonymous-4FAD/LightGBM'

_ABLATION_MODELS = {
    'lstm-cgm':               ('lstm', 'cgm'),
    'lstm-cgm-insulin':       ('lstm', 'cgm_insulin'),
    'lstm-cgm-carbs':         ('lstm', 'cgm_carbs'),
    'lstm-cgm-insulin-carbs': ('lstm', 'cgm_insulin_carbs'),
    'units-cgm':               ('units', 'cgm'),
    'units-cgm-insulin':       ('units', 'cgm_insulin'),
    'units-cgm-carbs':         ('units', 'cgm_carbs'),
    'units-cgm-insulin-carbs': ('units', 'cgm_insulin_carbs'),
    'gluforecast-cgm':               ('gluforecast', 'cgm'),
    'gluforecast-cgm-insulin':       ('gluforecast', 'cgm_insulin'),
    'gluforecast-cgm-carbs':         ('gluforecast', 'cgm_carbs'),
    'gluforecast-cgm-insulin-carbs': ('gluforecast', 'cgm_insulin_carbs'),
}


def _split_ablation(name: str, prefix: str) -> str:
    """Return ``ablation`` from ``<prefix>`` or ``<prefix>-<ablation>``."""
    if name == prefix:
        return 'all'
    return name[len(prefix) + 1:]  # strip "<prefix>-"


def get_model(name, device='cpu'):
    if name == 'gluformer':
        return Gluformer('anonymous-4FAD/Gluformer', device)
    elif name == 'gluformer-tiny':
        return Gluformer('anonymous-4FAD/Gluformer-tiny', device)
    elif name == 'zoh':
        return ZeroOrderHold()
    elif name == 'le':
        return LinearExtrapolation(15)
    elif name == 'ridge' or name.startswith('ridge-'):
        from models.ridge import Ridge as RidgeRunner
        return RidgeRunner(
            huggingface_model_name='anonymous-4FAD/Ridge',
            ablation=_split_ablation(name, 'ridge'),
            device=device,
        )
    elif name == 'lightgbm' or name.startswith('lightgbm-'):
        from models.lightgbm import LightGBM as LightGBMRunner
        # Bare ``lightgbm`` uses the repo default ablation from config.json.
        ablation = None if name == 'lightgbm' else _split_ablation(name, 'lightgbm')
        return LightGBMRunner(
            huggingface_model_name='anonymous-4FAD/LightGBM',
            ablation=ablation,
            device=device,
        )
    elif name in {'lstm', 'units', 'gluforecast'}:
        name = f'{name}-cgm-insulin-carbs'

    if name in _ABLATION_MODELS:
        model_type, feature_set = _ABLATION_MODELS[name]
        ckpt = f'{ABLATION_CKPT_DIR}/{model_type}_{feature_set}.pth'
        if model_type == 'lstm':
            return LSTM(ckpt, feature_set=feature_set, device=device)
        elif model_type == 'units':
            return UniTS(ckpt, feature_set=feature_set, device=device)
        else:
            return GluForecast(ckpt, feature_set=feature_set, device=device)
    else:
        raise ValueError(f'Model {name} not found')
