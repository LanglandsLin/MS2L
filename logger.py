import numpy as np

class Log:
    def __init__(self) -> None:
        self.batch_data = dict()
        self.epoch_data = dict()

    def update_batch(self, name, value):
        if name not in self.batch_data:
            self.batch_data[name] =  list()
        self.batch_data[name].append(value)

    def update_epoch(self):
        for name in self.batch_data.keys():
            if name not in self.epoch_data:
                self.epoch_data[name] = list()
            self.epoch_data[name].append(np.mean(self.batch_data[name]))
            self.batch_data[name] = list()
            print("{}: {}".format(name, self.epoch_data[name][-1]))
