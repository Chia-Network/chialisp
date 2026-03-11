import sys
import chialisp

this_module = sys.modules[__name__]
for key in dir(chialisp):
    setattr(this_module, key, getattr(chialisp, key))
