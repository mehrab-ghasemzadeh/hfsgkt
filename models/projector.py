from torch import nn

class Projector(nn.Linear):
    def __init__(self,d=512):
        super().__init__(d,d)
