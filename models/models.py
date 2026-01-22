from models.gluformer import Gluformer
from models.zoh import ZeroOrderHold
from models.linear import LinearRegression
from models.lstm import LSTM
from models.UniTS import UniTS
from models.glucose_decoder import GlucoseDecoderModel
from models.mean_regression import MeanRegression

def get_model(name):
    if name == 'gluformer':
        return Gluformer('njeffrie/Gluformer')
    elif name == 'gluformer-tiny':
        return Gluformer('njeffrie/Gluformer-tiny')
    elif name == 'zoh':
        return ZeroOrderHold()
    elif name == 'linear':
        return LinearRegression(15)
    elif name == 'lstm':
        return LSTM('njeffrie/LSTMGlucosePrediction')
    elif name == 'mean_regression':
        return MeanRegression()
    elif name == 'units':
        return UniTS('checkpoints/units_x128_prompt_tuning_checkpoint1.pth')
    elif name == 'glucose_decoder':
        return GlucoseDecoderModel(model_path='/Users/mkhvalchik/stanford/glucose_transformer/checkpoints/glucose_decoder_best.pth')
    else:
        raise ValueError(f'Model {name} not found')
