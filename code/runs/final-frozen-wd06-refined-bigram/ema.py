"""Training-only exponential moving average; inference uses a single model."""
import copy

import torch


class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # parameters() deduplicates tied embedding/output weights, so each
        # shared tensor receives exactly one EMA update.
        for average, current in zip(self.model.parameters(), model.parameters()):
            average.lerp_(current.detach(), 1.0 - self.decay)
        for average, current in zip(self.model.buffers(), model.buffers()):
            average.copy_(current)
        self.updates += 1
