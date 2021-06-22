import json

import numpy as np


class JsonObjectEncoder(json.JSONEncoder):

    np_int_dtypes = (np.int, np.int0, np.int8, np.int16, np.int32, np.int64, np.int_, np.intc,
                     np.uint, np.uint0, np.uint8, np.uint16, np.uint32, np.uint64, np.uintc,
                     np.uintp)

    np_float_dtypes = (np.float, np.float_, np.float16, np.float32, np.float64, np.float128)

    def default(self, obj):
        if isinstance(obj, self.np_int_dtypes):
            return int(obj)
        elif isinstance(obj, self.np_float_dtypes):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return list(obj)
        else:
            return json.JSONEncoder.default(self, obj)