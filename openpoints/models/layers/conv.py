from distutils.command.build import build
import torch.nn as nn
import logging
from .activation import create_act
from .norm import create_norm, create_norm

CONV_COUNT_1D = 0
CONV_COUNT_2D = 0
LINEAR_COUNT = 0

class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if len(args) == 2 and 'kernel_size' not in kwargs.keys():
            super(Conv2d, self).__init__(*args, (1, 1), **kwargs)
        else:
            super(Conv2d, self).__init__(*args, **kwargs)

    def forward(self, x):
        # global CONV_COUNT_2D
        # logging.info(f'Conv2d-{CONV_COUNT_2D}: {x.shape}')
        # CONV_COUNT_2D += 1
        return super(Conv2d, self).forward(x)


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        if len(args) == 2 and 'kernel_size' not in kwargs.keys():
            super(Conv1d, self).__init__(*args, 1, **kwargs)
        else:
            super(Conv1d, self).__init__(*args, **kwargs)
    
    def forward(self, x):
        # global CONV_COUNT_1D
        # logging.info(f'Conv1d-{CONV_COUNT_1D}: {x.shape}')
        # CONV_COUNT_1D += 1
        return super(Conv1d, self).forward(x)


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(*args, **kwargs)

    def forward(self, x):
        # global LINEAR_COUNT
        # logging.info(f'Linear-{LINEAR_COUNT}: {x.shape}')
        # LINEAR_COUNT += 1
        return super(Linear, self).forward(x)
    
    
def create_convblock2d(*args,
                       norm_args=None, act_args=None, order='conv-norm-act', **kwargs):
    in_channels = args[0]
    out_channels = args[1]
    bias = kwargs.pop('bias', True)
    if order == 'conv-norm-act':
        norm_layer = create_norm(norm_args, out_channels, dimension='2d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv2d(*args, bias=bias, **kwargs)]
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)

    elif order == 'norm-act-conv':
        conv_layer = []
        norm_layer = create_norm(norm_args, in_channels, dimension='2d')
        bias = False if norm_layer is not None else bias
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        conv_layer.append(Conv2d(*args, bias=bias, **kwargs))

    elif order == 'conv-act-norm':
        norm_layer = create_norm(norm_args, out_channels, dimension='2d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv2d(*args, bias=bias, **kwargs)]
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        if norm_layer is not None:
            conv_layer.append(norm_layer)
    else:
        raise NotImplementedError(f"{order} is not supported")

    return nn.Sequential(*conv_layer)


def create_convblock1d(*args,
                       norm_args=None, act_args=None, order='conv-norm-act', **kwargs):
    out_channels = args[1]
    in_channels = args[0]
    bias = kwargs.pop('bias', True)
    if order == 'conv-norm-act':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv1d(*args, bias=bias, **kwargs)]
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)

    elif order == 'norm-act-conv':
        conv_layer = []
        norm_layer = create_norm(norm_args, in_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        conv_layer.append(Conv1d(*args, bias=bias, **kwargs))

    elif order == 'conv-act-norm':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv1d(*args, bias=bias, **kwargs)]
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        if norm_layer is not None:
            conv_layer.append(norm_layer)
    else:
        raise NotImplementedError(f"{order} is not supported")

    return nn.Sequential(*conv_layer)


def create_linearblock(*args,
                       norm_args=None,
                       act_args=None,
                       order='conv-norm-act',
                       **kwargs):
    in_channels = args[0]
    out_channels = args[1]
    bias = kwargs.pop('bias', True)

    if order == 'conv-norm-act':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        linear_layer = [Linear(*args, bias, **kwargs)]
        if norm_layer is not None:
            linear_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            linear_layer.append(act_layer)
    elif order == 'norm-act-conv':
        linear_layer = []
        norm_layer = create_norm(norm_args, in_channels, dimension='1d')
        bias = kwargs.pop('bias', True)
        bias = False if norm_layer is not None else bias
        if norm_layer is not None:
            linear_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            linear_layer.append(act_layer)
        linear_layer.append(Linear(*args, bias=bias, **kwargs))

    elif order == 'conv-act-norm':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        linear_layer = [Linear(*args, bias=bias, **kwargs)]
        act_layer = create_act(act_args)
        if act_args is not None:
            linear_layer.append(act_layer)
        if norm_layer is not None:
            linear_layer.append(norm_layer)

    return nn.Sequential(*linear_layer)


class CreateResConvBlock2D(nn.Module):
    def __init__(self, mlps,
                 norm_args=None, act_args=None, order='conv-norm-act', **kwargs):
        super().__init__()
        self.convs = nn.Sequential()
        for i in range(len(mlps) - 2):
            self.convs.add_module(f'conv{i}',
                                  create_convblock2d(mlps[i], mlps[i + 1],
                                                     norm_args=norm_args, act_args=act_args, order=order, **kwargs))
        self.convs.add_module(f'conv{len(mlps) - 1}',
                              create_convblock2d(mlps[-2], mlps[-1], norm_args=norm_args, act_args=None, **kwargs))

        self.act = create_act(act_args)

    def forward(self, x, res=None):
        if res is None:
            return self.act(self.convs(x) + x)
        else:
            return self.act(self.convs(x) + res)
