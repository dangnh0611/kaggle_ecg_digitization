from torch import nn


class BaseModel(nn.Module):

    @property
    def state_dict_exclude_prefixes(self):
        return []
