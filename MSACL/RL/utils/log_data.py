"""Used in off_serial_trainer.py for sampling logging"""

from typing import Sequence, Union

class LogData:
    """Class to calculate and store running averages for logging"""
    def __init__(self):
        self.data = {}  # Stored average values
        self.counter = {}  # Count of updates per key

    def add_average(self, d: Union[dict, Sequence[dict]]):
        """Accumulate data and compute running average (dict/sequence of dicts)"""
        def _add_average(d: dict):
            for k, v in d.items():
                if k not in self.data:
                    self.data[k] = v
                    self.counter[k] = 1
                else:
                    self.data[k] = (self.data[k] * self.counter[k] + v) / (self.counter[k] + 1)
                    self.counter[k] += 1

        if isinstance(d, dict):
            _add_average(d)
        elif isinstance(d, Sequence):
            for di in d:
                _add_average(di)
        else:
            raise TypeError(f'Unsupported type {type(d)} for add_average!')

    def pop(self) -> dict:
        """Return averaged data and reset internal state"""
        data = self.data.copy()
        self.data = {}
        self.counter = {}
        return data
