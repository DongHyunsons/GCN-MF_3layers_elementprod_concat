from layers import *
import numpy as np
from inits import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


flags = tf.app.flags
FLAGS = flags.FLAGS


class SModel(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.user_layers = []
        self.item_layers = []
        
        self.user_layersConcat = []
        self.item_layersConcat = []
        
        self.user_activations = []
        self.itempos_activations = []
        self.userConcat_activations = []
        self.itemposConcat_activations = []        
        self.itemneg_activations = []
        
        
        
        
        
        #
        # self.user_history = None
        # self.item_history = None

        self.user_inputs = None
        self.itempos_inputs = None
        self.itemneg_inputs = None

        self.user_outputs = None
        self.user_outputsConcat = None
        self.itempos_outputs = None
        self.itempos_outputsConCat = None
        self.itemneg_outputs = None
        self.outputs_result = None
        self.outputs_result_mid = None
        self.GCNconcat = None
        
        self.FinalConcat = None
        

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.train_op = None
        self.test_op = None

        self.weight_loss = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build_history()
            self._build()

        # Build sequential layer model
        self.user_activations.append(self.user_inputs)
        for layer in self.user_layers:
            hidden = layer(self.user_activations[-1])
            self.user_activations.append(hidden)
        self.user_outputs = self.user_activations[-1]

        self.itempos_activations.append(self.itempos_inputs)
        for layer in self.item_layers:
            hidden = layer(self.itempos_activations[-1])
            self.itempos_activations.append(hidden)
        self.itempos_outputs = self.itempos_activations[-1]

        # Build concat deep layers
        self.userConcat_activations.append(self.user_inputs)
        for layer in self.user_layersConcat:
            hidden = layer(self.userConcat_activations[-1])
            self.userConcat_activations.append(hidden)
        self.user_outputsConcat = self.userConcat_activations[-1]        

        self.itemposConcat_activations.append(self.itempos_inputs)
        for layer in self.item_layersConcat:
            hidden = layer(self.itemposConcat_activations[-1])
            self.itemposConcat_activations.append(hidden)
        self.itempos_outputsConCat = self.itemposConcat_activations[-1]        
        
        
        
        
        
        #
        self.update_history = []




        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = [self.optimizer.minimize(self.loss)]
        self.train_op = []
        with tf.control_dependencies(self.opt_op):
            self.train_op = tf.group(*self.update_history)
        self.test_op = tf.group(*self.update_history)

    def predict(self):
        pass

    def _build_history(self):
        pass


    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)




class SGCN_MFConcat(SModel): #mixture of dense and gcn
    def __init__(self, placeholders, user_length, item_length, user_input_dim, item_input_dim, **kwargs):
        super(SGCN_MFConcat, self).__init__(**kwargs)
        self.user_inputs = placeholders['user_AXfeatures']# A*X for the bottom layer, not original feature X
        self.itempos_inputs = placeholders['itempos_AXfeatures']

        self.labels = placeholders['labels']
        self.labelweight = placeholders['label_weights']

        self.user_length = user_length
        self.item_length = item_length
        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = FLAGS.embedding_size
        self.placeholders = placeholders
        #self.support = placeholders['support']
        #self.loss_decay = placeholders['loss_decay']

        #self.weight_loss = placeholders['weight_loss']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)



        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.user_layers[0].vars.values():
            self.loss += FLAGS.weight_decay  * tf.nn.l2_loss(var)

        for var in self.item_layers[0].vars.values():
            self.loss += FLAGS.weight_decay  * tf.nn.l2_loss(var)

        #self.outputs_result = tf.clip_by_value(tf.matmul(self.user_outputs, self.itempos_outputs, transpose_b=True), 0, 1) ##### 이부분이 원래 코드
        
        
        tmpA_shallow = tf.repeat(self.itempos_outputs[tf.newaxis,:,:],3209,axis=0)                     #### 밑에 Matmul  빼고 concat을 해보자
        tmpB_shallow = tf.repeat(self.user_outputs[:,tf.newaxis,:],12331,axis=1)
        
        #self.outputs_result_mid = tf.concat([tmpA_shallow,tmpB_shallow] , axis = 2)   ############################# 
                                                                                      ####################### 여기부분에서 elementwise product로 바꿔보자
        
        ########################################## elementwise producrt#########################################
        self.outputs_result_mid = tf.multiply(tmpA_shallow , tmpB_shallow)
        
        
        #self.outputs_result_mid = tf.matmul(self.user_outputs, self.itempos_outputs , transpose_b = True)  #### clip func 없애보기          
        #self.outputs_result_mid = tf.repeat(self.outputs_result_mid[:,:,tf.newaxis],1,axis=2)              #######  기존 layer에서 concat을 위해 리쉐이프
        
        
        tmpA = tf.repeat(self.itempos_outputsConCat[tf.newaxis,:,:],3209,axis=0)                     #### ConcatDeepLayer에서 concat을 위해 reshape
        tmpB = tf.repeat(self.user_outputsConcat[:,tf.newaxis,:],12331,axis=1)
        
        
        
        self.GCNconcat = tf.concat([tmpA,tmpB] , axis = 2)         ########### GCN layer concat
        
        init_range = np.sqrt(6.0/40)
        initial_deeplayers1 = tf.random_uniform([40,40], minval = -init_range, maxval = init_range, dtype = tf.float32)

        self.vars['deeplayers1'] = tf.Variable(initial_deeplayers1)
        self.vars['deeplayers2'] = tf.Variable(initial_deeplayers1)
        self.vars['deeplayers3'] = tf.Variable(initial_deeplayers1)
        self.vars['deeplayers4'] = tf.Variable(initial_deeplayers1)

        self.GCN_deeplayer1 = tf.nn.relu(tf.matmul(self.GCNconcat, self.vars['deeplayers1'], transpose_b = True))
        self.GCN_deeplayer2 = tf.nn.relu(tf.matmul(self.GCN_deeplayer1, self.vars['deeplayers1'], transpose_b = True))
        self.GCN_deeplayer3 = tf.nn.relu(tf.matmul(self.GCN_deeplayer2, self.vars['deeplayers1'], transpose_b = True))
        self.GCN_deeplayer4 = tf.nn.relu(tf.matmul(self.GCN_deeplayer3, self.vars['deeplayers1'], transpose_b = True))




        
        
        
        self.FinalConcat = tf.concat([self.outputs_result_mid,self.GCN_deeplayer4] , axis = 2)                ##### elementproduct output과 gcnconcatlayer concat
        #self.FinalConcat = tf.concat([self.outputs_result_mid,self.outputs_result_mid] , axis = 2)    # 위에코드 메모리부족해서 임시로 대체했던 코드
        
        
        init_range = np.sqrt(6.0/(240))
        initial = tf.random_uniform([1,240], minval=-init_range, maxval=init_range, dtype=tf.float32)
        #init_range = np.sqrt(6.0/(2))
        #initial = tf.random_uniform([1,2], minval=-init_range, maxval=init_range, dtype=tf.float32)
        
        
        self.vars['finalNeural'] = tf.Variable(initial)
        
        
        self.outputs_result = tf.matmul(self.FinalConcat,self.vars['finalNeural'],transpose_b = True)
        
        
        tmplabels = tf.repeat(self.labels[:,:,tf.newaxis],1,axis=2)
        tmplabelweight = tf.repeat(self.labelweight[:,:,tf.newaxis],1,axis=2)
        
        
        
        
        self.loss += tf.reduce_sum(
            tf.multiply(tf.squared_difference(self.outputs_result, tmplabels), tmplabelweight))
        #self.outputs_result = tf.reduce_sum(tf.multiply(self.user_outputs, (self.itempos_outputs - self.itemneg_outputs)), 1, keep_dims=True)

        #self.loss += -tf.reduce_mean(tf.log(tf.sigmoid(self.outputs_result)))
        #self.loss += - tf.reduce_mean(tf.multiply(tf.log(tf.sigmoid(self.outputs_result)), self.weight_loss))



    def _accuracy(self):
        self.accuracy = tf.reduce_mean(tf.to_float(self.outputs_result > 0))

    def _build(self):
        self.user_layers.append(InductiveUser(input_dim=self.user_input_dim,
                                 #output_dim=FLAGS.user_hidden1,
                                 output_dim=200,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))


        self.item_layers.append(InductiveItem(input_dim=self.item_input_dim,
                                 #output_dim=FLAGS.item_hidden1,
                                 output_dim=200,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,
                                 logging=self.logging))
        
        
        self.user_layersConcat.append(InductiveUserConcat(input_dim=self.user_input_dim,
                                 #output_dim=FLAGS.item_hidden1,
                                 output_dim=FLAGS.user_hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=0.3,
                                 sparse_inputs=False,
                                 logging=self.logging))
        
        self.item_layersConcat.append(InductiveItemConcat(input_dim=self.item_input_dim,
                                 #output_dim=FLAGS.item_hidden1,
                                 output_dim=FLAGS.user_hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=0.3,
                                 sparse_inputs=False,
                                 logging=self.logging))
        
        
        


    def predict(self):
        return tf.nn.softmax(self.outputs)

    def _build_history(self):
        # Create history after each aggregation
        return



