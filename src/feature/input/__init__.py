from collections import namedtuple
from typing import List, Union

import torch.nn as nn


class Inputs(nn.Module):
    r"""General Input class.
    """

    def __init__(self):
        # refer to parent class
        super(Inputs, self).__init__()

    def __len__(self) -> int:
        r"""Return outputs size.
        
        Returns:
            int: Size of embedding tensor, or Number of inputs' fields.
        """
        return self.length

    def set_schema(self, inputs: Union[str, List[str]]):
        r"""Initialize input layer's schema.
        
        Args:
            inputs (Union[str, List[str]]): String or list of strings of inputs' field names.
        """
        # convert string to list of string
        if isinstance(inputs, str):
            inputs = [inputs]

        # create a namedtuple of schema
        schema = namedtuple("Schema", ["inputs"])

        # initialize self.schema with the namedtuple
        self.schema = schema(inputs=inputs)