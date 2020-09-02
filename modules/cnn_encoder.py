from typing import Tuple, Optional

import torch
from torch.nn import Module, Conv2d, Linear, ModuleList

from allennlp.nn import Activation


class CnnDialogueEncoder(Module):
    """
    A `CNNDialogueEncoder` encoding dialogue (seq of seq) directly. It is inspired by AllenNLP CnnEncoder and
    "Convolutional Neural Network for Sentence Classification, Y Kim 2014.
    A `CnnEncoder` is a combination of multiple convolution layers and max pooling layers.
    The input to this module is of shape `(batch_size, num_sentences, num_tokens, input_dim)`,
    and the output is of shape `(batch_size, output_dim)`.
    The CNN has one convolution layer for each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    is `num_tokens - ngram_size + 1`. The corresponding maxpooling layer aggregates all these
    outputs from the convolution layer and outputs the max.
    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is `len(ngram_filter_sizes) * num_filters`.  This then gets
    (optionally) projected down to a lower dimensional output, specified by `output_dim`.
    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.
    # Parameters
    embedding_dim : `int`, required
        This is the input dimension to the encoder.  We need this because we can't do shape
        inference in pytorch, and we need to know what size filters to construct in the CNN.
    num_filters : `int`, required
        This is the output dim for each convolutional layer, which is the number of "filters"
        learned by that layer.
    ngram_filter_sizes : `Tuple[int]`, optional (default=`(2, 3, 4, 5)`)
        This specifies both the number of convolutional layers we will create and their sizes.  The
        default of `(2, 3, 4, 5)` will have four convolutional layers, corresponding to encoding
        ngrams of size 2 to 5 with some number of filters.
    conv_layer_activation : `Activation`, optional (default=`torch.nn.ReLU`)
        Activation to use after the convolution layers.
    output_dim : `Optional[int]`, optional (default=`None`)
        After doing convolutions and pooling, we'll project the collected features into a vector of
        this size.  If this value is `None`, we will just return the result of the max pooling,
        giving an output of shape `len(ngram_filter_sizes) * num_filters`.
    """

    def __init__(self,
                 embedding_dim: int = None,
                 num_filters: int = None,
                 ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),
                 conv_layer_activation: Activation = None,
                 output_dim: Optional[int] = None
                 ) -> None:
        super(CnnDialogueEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = conv_layer_activation or Activation.by_name("relu")()
        self._output_dim = output_dim

        self._convolution_layers = ModuleList(
            [
            Conv2d(in_channels=self._embedding_dim,
                   out_channels=self._num_filters,
                   kernel_size=ngram_size)
            for ngram_size in self._ngram_filter_sizes
            ]
        )
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module("conv_layer_%d" % i, conv_layer)

        maxpool_output_dim = self._num_filters * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = Linear(maxpool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self,
                sentences: torch.Tensor):
        # Input is expected to have shape `(batch_size, num_sentences, num_tokens, embedding_dim)`.
        # The convolution layers expect input of shape `(batch_size, in_channles, sequence_lenth, sequence_lenth)`,
        # where the conv layer `in_channels` is `embedding_dim`.
        # Thus it needs to transpose(permute) the tensor first.
        sentences = sentences.permute(0, 3, 1, 2)
        # Each convolution layer returns output of size `(batch_size, num_filters, pool_length1, pool_length2)`,
        # where `pool_length1 = num_sentences - ngram_size + 1` and `pool_length2 = num_tokens - ngram_size + 1`.
        # We then do an activation function,
        # then do max pooling over each filter for the whole input sequence.  Because our max
        # pooling is simple, we just use `torch.max`.  The resultant tensor of has shape
        # `(batch_size, num_conv_layers * num_filters)`, which then gets projected using the
        # projection layer, if requested.

        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, "conv_layer_{}".format(i))
            output = self._activation(convolution_layer(sentences))
            filter_outputs.append(output.view(output.size(0), output.size(1), -1).max(dim=-1)[0])

        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = (
            torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]
        )

        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result