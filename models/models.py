from models.gluformer import Gluformer

model_name_map = {'gluformer': Gluformer}

def get_model(model_name: str):
    if model_name in model_name_map:
        return model_name_map[model_name]()
    else:
        raise ValueError(f'Model {model_name} not found')