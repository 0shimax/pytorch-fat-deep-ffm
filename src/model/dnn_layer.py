from typing import Callable, List

import torch
import torch.nn as nn


class MultilayerPerceptionLayer(nn.Module):
    r"""Layer class of Multilayer Perception (MLP), which is also called fully connected
    layer, dense layer, deep neural network, etc, to calculate high order non linear 
    relations of features with a stack of linear, dropout and activation.
    """

    def __init__(self,
                 inputs_size: int,
                 output_size: int,
                 layer_sizes: List[int],
                 dropout_p: List[float] = None,
                 activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        """Initialize MultilayerPerceptionLayer
        
        Args:
            inputs_size (int): Input size of MLP, i.e. size of embedding tensor. 
            output_size (int): Output size of MLP
            layer_sizes (List[int]): Layer sizes of MLP
            dropout_p (List[float], optional): Probability of Dropout in MLP. 
                Defaults to None.
            activation (Callable[[T], T], optional): Activation function in MLP. 
                Defaults to nn.ReLU().
        
        Attributes:
            inputs_size (int): Input size of MLP. 
            model (torch.nn.Sequential): Sequential of MLP.
        
        Raises: ValueError: when embed_size or num_fields is missing if using embed_size and num_field pairs,
        or when inputs_size is missing if using inputs_size ValueError: when dropout_p is not None and length of
        dropout_p is not equal to that of layer_sizes
        """
        # refer to parent class
        super(MultilayerPerceptionLayer, self).__init__()

        # check if length of dropout_p is not equal to length of layer_sizes
        if dropout_p is not None and len(dropout_p) != len(layer_sizes):
            raise ValueError("length of dropout_p must be equal to length of layer_sizes.")

        # bind inputs_size to inputs_size
        self.inputs_size = inputs_size

        # create a list of inputs_size and layer_sizes
        layer_sizes = [inputs_size] + layer_sizes

        layers = []
        # initialize module of linear, activation and dropout, and add them to sequential module
        for i, (in_f, out_f) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(in_f, out_f))
            if activation is not None:
                layers.append(activation)
            if dropout_p is not None:
                layers.append(nn.Dropout(dropout_p[i]))

        # initialize module of linear and add it to sequential module
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        # initialize sequential of model
        self.model = nn.Sequential(*layers)

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of MultilayerPerceptionLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, N, O), dtype = torch.float: Output of MLP.
        """
        # Calculate with model forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: outputs, shape = (B, N, O)
        outputs = self.model(emb_inputs.rename(None))

        # Rename tensor names
        if outputs.dim() == 2:
            outputs.names = ("B", "O")
        elif outputs.dim() == 3:
            outputs.names = ("B", "N", "O")

        return outputs