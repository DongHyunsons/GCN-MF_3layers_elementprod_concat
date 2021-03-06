from inits import *
import tensorflow as tf


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



flags = tf.app.flags
FLAGS = flags.FLAGS


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                if isinstance(inputs,tuple):
                    tf.summary.histogram(self.name + '/inputs', inputs[0])
                else:
                    tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                if isinstance(outputs,tuple):
                    tf.summary.histogram(self.name + '/outputs', outputs[0])
                else:
                    tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


            
            
            
            
            
            
            
            
            
            
class InductiveUser(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(InductiveUser, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class InductiveItem(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(InductiveItem, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        #output = tf.nn.l2_normalize(output,dim=0)
        return self.act(output)

    

class InductiveUserConcat(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.3, sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(InductiveUserConcat, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            self.vars['weights2'] = glorot([output_dim, output_dim],
                                          name='weights2')
            self.vars['weights3'] = glorot([output_dim, 20],
                                          name='weights3')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
                self.vars['bias2'] = zeros([output_dim], name='bias2')
                self.vars['bias3'] = zeros([output_dim], name='bias3')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        output2 =  self.act(output)
        
        
        
        ####################2nd layer
         # dropout
        if self.sparse_inputs:
            output2 = sparse_dropout(output2, 1-self.dropout, self.num_features_nonzero)
        else:
            output2 = tf.nn.dropout(output2, 1-self.dropout)        
        
        # transform
        output3 = dot(output2, self.vars['weights2'], sparse=self.sparse_inputs)
        
        # bias
        if self.bias:
            output3 += self.vars['bias2']
        
        output4 = self.act(output3)
    
        ####################3rd layer
         # dropout
        if self.sparse_inputs:
            output4 = sparse_dropout(output4, 1-self.dropout, self.num_features_nonzero)
        else:
            output4 = tf.nn.dropout(output4, 1-self.dropout)        
        
        # transform
        output5 = dot(output4, self.vars['weights3'], sparse=self.sparse_inputs)
        
        # bias
        if self.bias:
            output5 += self.vars['bias3']
        
        return self.act(output5)
        
        

class InductiveItemConcat(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.3, sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(InductiveItemConcat, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            self.vars['weights2'] = glorot([output_dim, output_dim],
                                          name='weights2')
            self.vars['weights3'] = glorot([output_dim, 20],
                                          name='weights3')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')
                self.vars['bias2'] = zeros([output_dim], name='bias2')
                self.vars['bias3'] = zeros([output_dim], name='bias3')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        output2 =  self.act(output)
        
        
        
        
        ####################2nd layer
         # dropout
        if self.sparse_inputs:
            output2 = sparse_dropout(output2, 1-self.dropout, self.num_features_nonzero)
        else:
            output2 = tf.nn.dropout(output2, 1-self.dropout)        
        
        # transform
        output3 = dot(output2, self.vars['weights2'], sparse=self.sparse_inputs)
        
        # bias
        if self.bias:
            output3 += self.vars['bias2']
        
        output4 = self.act(output3)
    
        ####################3rd layer
         # dropout
        if self.sparse_inputs:
            output4 = sparse_dropout(output4, 1-self.dropout, self.num_features_nonzero)
        else:
            output4 = tf.nn.dropout(output4, 1-self.dropout)        
        
        # transform
        output5 = dot(output4, self.vars['weights3'], sparse=self.sparse_inputs)
        
        # bias
        if self.bias:
            output5 += self.vars['bias3']
        
        return self.act(output5)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class GraphConvolution_Flag(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 support=None, sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, flag = None, **kwargs):
        super(GraphConvolution_Flag, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        if support is None:
            self.support = placeholders['support']
        else:
            self.support = support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.save_output = None
        self.flag = flag

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(1):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero[self.flag])
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # supports = list()
        # for i in range(len(self.support)):
        #     if not self.featureless:
        #         pre_sup = dot(x, self.vars['weights_' + str(i)],
        #                       sparse=self.sparse_inputs)
        #     else:
        #         pre_sup = self.vars['weights_' + str(i)]
        #     support = dot(self.support[i], pre_sup, sparse=True)
        #     supports.append(support)
        # output = tf.add_n(supports)
        if not self.featureless:
            pre_sup = dot(x, self.vars['weights_0'],
                          sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights_0']
        output = dot(self.support[self.flag], pre_sup, sparse=True)

        # bias
        if self.bias:
            output += self.vars['bias']
        output = tf.nn.l2_normalize(output,dim=0)
        output = self.act(output)
        self.save_output = output
        return output
