# import os
# __all__ = [
#     filename[:-3]
#     for filename in os.listdir(os.path.dirname(__file__))
#     if filename[-3:]=='.py' and filename[:-3]!='__init__'
# ]
# del os
# __all__ = ['path', 'dataloader', 'model', 'optimizer', 'training']

from . import *
from .config_local import *
from .path import *
from .config_local import *
from .dataloader import *
from .optimizer import *
from .model import *

dict_Params = {
    key:var
    for key, var in globals().items()
    if key.startswith('params_')
}
def getParams():
    return dict_Params
lst_Params = list(dict_Params.keys())

dict_Paths = {
    key:var
    for key, var in globals().items()
    if key.startswith('path_')
}
def getPaths():
    return dict_Paths
lst_Paths = list(dict_Paths.keys())

dict_Config = {}
dict_Config.update(dict_Paths)
dict_Config.update(dict_Params)
def getConfigs():
    return dict_Config

__all__ = []
__all__.extend(lst_Paths)
__all__.extend(lst_Params)
