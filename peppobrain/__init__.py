def _import_nn():
    import peppobrain.nn
    return peppobrain.nn

def _import_utils():
    import peppobrain.utils
    return peppobrain.utils

nn = _import_nn()
utils = _import_utils()