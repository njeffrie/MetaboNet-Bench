from models.gluformer import Gluformer
from models.zoh import ZeroOrderHold
from models.linear import LinearExtrapolation
from models.lstm import LSTM
from models.UniTS import UniTS
from models.gluforecast import GluForecast

ABLATION_CKPT_DIR = 'train/checkpoints'

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


def get_model(name, device='cpu'):
    if name == 'gluformer':
        return Gluformer('njeffrie/Gluformer', device)
    elif name == 'gluformer-tiny':
        return Gluformer('njeffrie/Gluformer-tiny', device)
    elif name == 'zoh':
        return ZeroOrderHold()
    elif name == 'le':
        return LinearExtrapolation(15)
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
