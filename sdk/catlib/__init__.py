from os.path import realpath, dirname, join as joinpath
SDK_BASEPATH = dirname(dirname(realpath(__file__)))
SCRATCH_BASEPATH = joinpath(SDK_BASEPATH, 'scratch')
from .cat_db import CatDB