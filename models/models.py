from models.gluformer import Gluformer
from models.zoh import ZeroOrderHold
from models.linear import LinearRegression
from models.lstm import LSTM
from models.UniTS import UniTS

model_name_map = {'gluformer': Gluformer('njeffrie/Gluformer'),
                  'gluformer-tiny': Gluformer('njeffrie/Gluformer-tiny'),
                  'zoh': ZeroOrderHold(),
                  'linear': LinearRegression(15),
                  'lstm': LSTM('njeffrie/LSTMGlucosePrediction'),
                  'units': UniTS('checkpoints/units_x128_pretrain_checkpoint.pth')}

def get_model(model_name: str):
    if model_name in model_name_map:
        return model_name_map[model_name]
    else:
        raise ValueError(f'Model {model_name} not found')
