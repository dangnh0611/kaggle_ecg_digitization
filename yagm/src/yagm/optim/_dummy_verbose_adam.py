import torch


class _DummyVerboseAdam(torch.optim.Adam):
    """For debuging purpose only
    """
    def step(self, closure=None):
        print('Start optimizer step with LR', self.param_groups[0]["lr"])
        ret = super().step(closure)
        print('Done optimizer step with LR', self.param_groups[0]["lr"])
        return ret