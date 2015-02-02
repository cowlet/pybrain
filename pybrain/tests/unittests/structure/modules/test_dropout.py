"""

    >>> from pybrain.structure import ReluLayer
    >>> from scipy import random, array, empty

Set the random seed so we can predict the random variables.
randint gives
[0, 1, 1]
[0, 1, 1]
[1, 1, 1]
[1, 1, 0]

    >>> random.seed(0)

Create a layer.

    >>> layer = ReluLayer(3)
    >>> layer.dropout = True
    >>> input = array((0.5, 0.5, 1.))
    >>> output = empty((3,))

Now test some forwards:

    >>> output = layer.activate(input)
    >>> output
    array([ 0. ,  0.5, 1. ])

    >>> output = layer.activate(input)
    >>> output
    array([ 0. ,  0.5, 1. ])

    >>> output = layer.activate(input)
    >>> output
    array([ 0.5,  0.5, 1. ])

    >>> output = layer.activate(input)
    >>> output
    array([ 0.5,  0.5, 0. ])

And backwards:

    >>> outerr = array((0.1, 0.2, 0.3))
    >>> inerr = layer.backActivate(outerr)
    >>> inerr
    array([ 0.1,  0.2, 0. ])

"""

__author__ = 'Victoria Catterson, vic@cowlet.org'

from pybrain.tests import runModuleTestSuite

if __name__ == "__main__":
    runModuleTestSuite(__import__('__main__'))

