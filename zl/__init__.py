# trying to finish a learning package as a whole
# starting at 17.10.06

from . import utils
from .backends import BK

def init_all(opts):
    utils.init(opts["log"])
    BK.init(opts)
