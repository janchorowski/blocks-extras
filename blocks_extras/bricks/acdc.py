import theano
import numpy

from theano import tensor
from theano.sandbox.cuda.acdc import FastACDC

from blocks.bricks import Feedforward, Initializable, Rectifier, Bias
from blocks.bricks.base import lazy, application
from blocks.initialization import Constant
from blocks.roles import add_role, WEIGHT, BIAS, ParameterRole
from blocks.utils import shared_floatx_nans


class FixedParameterRole(ParameterRole):
    pass

#: Parameters that are not modified during training
FIXED = FixedParameterRole()


class ACDC(Feedforward, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, **kwargs):
        super(ACDC, self).__init__(**kwargs)
        self.dim = dim

    @property
    def input_dim(self):
        return self.dim

    @property
    def output_dim(self):
        return self.dim

    def _allocate(self):
        A = shared_floatx_nans((self.dim, ), name='A')
        add_role(A, WEIGHT)
        self.parameters.append(A)

        Ab = shared_floatx_nans((self.dim, ), name='Ab')
        add_role(Ab, BIAS)
        self.parameters.append(Ab)

        D = shared_floatx_nans((self.dim, ), name='D')
        add_role(D, WEIGHT)
        self.parameters.append(D)

        Db = shared_floatx_nans((self.dim, ), name='Db')
        add_role(Db, BIAS)
        self.parameters.append(Db)

    def _initialize(self):
        A, Ab, D, Db = self.parameters
        self.weights_init.initialize(A, self.rng)
        self.biases_init.initialize(Ab, self.rng)
        self.weights_init.initialize(D, self.rng)
        self.biases_init.initialize(Db, self.rng)

    @application
    def apply(self, x):
        A, Ab, D, Db = self.parameters
        return FastACDC()(x, A, Ab, D, Db)


class Permutation(Feedforward, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, **kwargs):
        super(Permutation, self).__init__(**kwargs)
        self.dim = dim

    @property
    def input_dim(self):
        return self.dim

    @property
    def output_dim(self):
        return self.dim

    def _allocate(self):
        P = theano.shared(numpy.zeros((self.dim,), dtype=numpy.int64),
                          name='P')
        add_role(P, FIXED)
        self.parameters.append(P)

    def _initialize(self):
        P, = self.parameters
        P.set_value(self.rng.permutation(self.dim).astype(numpy.int64))

    @application
    def apply(self, input_):
        P, = self.parameters
        return tensor.advanced_subtensor1(
            input_.dimshuffle(1, 0), P).dimshuffle(1, 0)


class ACDCStack(Initializable, Feedforward):
    @lazy(allocation=['dim'])
    def __init__(self, dim, num_acdcs=1, **kwargs):
        super(ACDCStack, self).__init__(**kwargs)
        self._dim = dim
        self.children.append(ACDC(dim=self.dim, name='ACDC0'))
        self.children.append(Rectifier(name='ReLU0'))
        for i in xrange(1, num_acdcs):
            self.children.append(Permutation(dim=self.dim,
                                             name='Perm%d' % (i,)))
            self.children.append(ACDC(dim=self.dim, name='ACDC%d' % (i,)))
            self.children.append(Rectifier(name='ReLU%d' % (i, )))
        self.children.append(Bias(dim=self.dim))

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value
        for c in self.children:
            c.dim = value

    @property
    def input_dim(self):
        return self.dim

    @input_dim.setter
    def input_dim(self, value):
        self.dim = value

    @property
    def output_dim(self):
        return self.dim

    @output_dim.setter
    def output_dim(self, value):
        self.dim = value

    def _push_initialization_config(self):
        bi = self.biases_init
        self.biases_init = Constant(0.0)
        super(ACDCStack, self)._push_initialization_config()
        self.biases_init = bi
        self.children[-1].biases_init = bi

    @application
    def apply(self, input_):
        for c in self.children:
            input_ = c.apply(input_)
        return input_
