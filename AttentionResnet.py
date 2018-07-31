import tensorflow as tf

Detection_or_Classifier = 'Classifier'#Detection,Classifier

class AttentionResnet():
    def __init__(self,num_classes=1000,learning_rate=0.01):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        self.__build()
    
    def __build(self):
        self.norm = 'batch_norm'#group_norm,batch_norm,None
        self.activate = 'prelu'#selu,leaky,swish,relu,relu6,prelu,None
        
        self.__init_global_epoch()
        self.__init_global_step()
        self.__init_input()
        
        with tf.variable_scope('zsc_preprocessing'):
            red, green, blue = tf.split(self.input_image, num_or_size_splits=3, axis=3)
            x = tf.concat([
                           tf.truediv(tf.subtract(tf.truediv(blue,255.0), 0.5),0.5),
                           tf.truediv(tf.subtract(tf.truediv(green,255.0), 0.5),0.5),
                           tf.truediv(tf.subtract(tf.truediv(red,255.0), 0.5),0.5),
                          ], 3)
        
        with tf.variable_scope('zsc_feature'):
            x = PrimaryModule('PrimaryModule',x,self.norm,self.activate,self.is_training)
            
            x = AttentionResBlock('AttentionResBlock_1',x,[64,64,256],1,3,self.norm,self.activate,self.is_training)
            
            x = AttentionResBlock('AttentionResBlock_2',x,[128,128,512],2,4,self.norm,self.activate,self.is_training)
            
            x = AttentionResBlock('AttentionResBlock_3',x,[256,256,1024],2,6,self.norm,self.activate,self.is_training)
            
            x = AttentionResBlock('AttentionResBlock_4',x,[512,512,2048],2,3,self.norm,self.activate,self.is_training)
            
        if Detection_or_Classifier=='Classifier':
            with tf.variable_scope('zsc_classifier'):
                x = tf.reduce_mean(x,[1,2],keep_dims=True)
                self.classifier_logits = tf.reshape(_conv_block('Logits',x,self.num_classes,1,1,'SAME',None,None,self.is_training),
                                                    [tf.shape(x)[0],self.num_classes])
        elif Detection_or_Classifier=='Detection':
            with tf.variable_scope('zsc_detection'):
                pass
        
        self.__init__output()
    
    def __init__output(self):
        with tf.variable_scope('output'):
            regularzation_loss = sum(tf.get_collection("regularzation_loss"))
            
            if Detection_or_Classifier=='Classifier':
                self.all_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.classifier_logits, labels=self.y, name='loss'))
                self.all_loss = self.all_loss + regularzation_loss
                
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    learning_rate = tf.train.exponential_decay(self.learning_rate,global_step=self.global_epoch_tensor,decay_steps=5,decay_rate=0.9995,staircase=True)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate)
                    self.train_op = self.optimizer.minimize(self.all_loss)
                
                self.y_out_softmax = tf.nn.softmax(self.classifier_logits,name='zsc_output')
                self.y_out_argmax = tf.cast(tf.argmax(self.y_out_softmax, axis=-1),tf.int32)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_out_argmax), tf.float32))
                self.accuracy_top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.y_out_softmax,self.y,5),tf.float32))

                #with tf.name_scope('train-summary-per-iteration'):
                #    tf.summary.scalar('loss', self.all_loss)
                #    tf.summary.scalar('acc', self.accuracy)
                #    tf.summary.scalar('acc', self.accuracy_top_5)
                #    self.summaries_merged = tf.summary.merge_all()
            elif Detection_or_Classifier=='Detection':
                pass
    def __init_input(self):
        if Detection_or_Classifier=='Classifier':
            with tf.variable_scope('input'):
                self.input_image = tf.placeholder(tf.float32,[None, None, None, 3],name='zsc_input')
                self.y = tf.placeholder(tf.int32, [None],name='zsc_input_target')
                self.is_training = tf.placeholder(tf.bool,name='zsc_is_train')
        elif Detection_or_Classifier=='Detection':
            with tf.variable_scope('input'):
                pass
    def __init_global_epoch(self):
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)
    def __init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

#################################################################################################################
#################################################################################################################
#################################################################################################################
###################################################LAYERS########################################################
##AttentionResBlock
def AttentionResBlock(name,x,num_filters=[64,64,256],stride=1,repeat=3,norm='group_norm',activate='selu',is_training=True):
    assert stride==1 or stride==2
    with tf.variable_scope(name):
        for i in range(repeat):
            if i==0:
                if stride==2:
                    x = ConvBlock('ConvBlock_{}'.format(i),x,num_filters,2,norm,activate,is_training)
                else:
                    x = ConvBlock('ConvBlock_{}'.format(i),x,num_filters,1,norm,activate,is_training)
            else:
                input = x
                x = ConvBlock('ConvBlock_{}'.format(i),x,num_filters,1,norm,activate,is_training)
                _selfattention = SelfAttention('SelfAttention_{}'.format(i),input,x,x,norm,activate,is_training)
                
                C = x.shape.as_list()[-1]
                SE_input = tf.reduce_mean(input,[1,2],keep_dims=True)
                SE_x = tf.reduce_mean(x,[1,2],keep_dims=True)
                SE = tf.concat([SE_input,SE_x],axis=-1)
                SE = _conv_block('SE_{}_1'.format(i),SE,SE.shape.as_list()[-1]//16,1,1,'SAME',None,activate,is_training)
                SE = _conv_block('SE_{}_2'.format(i),SE,C,1,1,'SAME',None,None,is_training)
                SE = tf.nn.sigmoid(SE)
                
                x = input + (x+_selfattention)*(1.0+SE)
            
        return x
def ConvBlock(name,x,num_filters=[64,64,256],stride=1,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        x = _conv_block('conv_0',x,num_filters[0],1,1,'SAME',norm,activate,is_training)
        x = _conv_block('conv_1',x,num_filters[1],3,stride,'SAME',norm,activate,is_training)
        x = _conv_block('conv_2',x,num_filters[2],1,1,'SAME',norm,activate,is_training)
        return x
##PrimaryModule
def PrimaryModule(name,x,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        x = _conv_block('conv_0',x,64,3,2,'SAME',norm,activate,is_training)
        x = _conv_block('conv_1',x,64,3,1,'SAME',norm,activate,is_training)
        x = tf.nn.max_pool(x,[1,3,3,1],[1,2,2,1],'SAME')
        return x
##selfattention
def SelfAttention(name,f,g,h,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        
        C = h.get_shape().as_list()[-1]
        f = _conv_block('f',f,C//8,1,1,'SAME',norm,activate,is_training)
        g = _conv_block('g',g,C//8,1,1,'SAME',norm,activate,is_training)
        h = _conv_block('h',h,C,1,1,'SAME',norm,activate,is_training)
        
        s = tf.matmul(tf.reshape(g,[tf.shape(g)[0],-1,tf.shape(g)[-1]]), tf.reshape(f,[tf.shape(f)[0],-1,tf.shape(f)[-1]]), transpose_b=True) # # [bs, N, N]

        beta = tf.nn.softmax(s, dim=-1)  # attention map

        o = tf.matmul(beta, tf.reshape(h,[tf.shape(h)[0],-1,tf.shape(h)[-1]])) # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, tf.concat([tf.shape(h)[:3],[-1]],axis=-1)) # [bs, h, w, C]
        x = gamma * o
        
        return x
##_conv_block
def _conv_block(name,input,num_filters=16,kernel_size=3,stride=2,padding='SAME',norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        w = GetWeight('weight',[kernel_size,kernel_size,input.shape.as_list()[-1],num_filters])
        x = tf.nn.conv2d(input,w,[1,stride,stride,1],padding=padding,name='conv')
        
        b = tf.get_variable('bias',num_filters,tf.float32,initializer=tf.constant_initializer(0.0))
        x += b
        
        if norm=='batch_norm':
            x = bn(x, is_training, name='batch_norm')
        elif norm=='group_norm':
            x = group_norm(x,name='groupnorm')
        else:
            pass
        
        if activate=='leaky': 
            x = LeakyRelu(x,leak=0.1, name='leaky')
        elif activate=='selu':
            x = selu(x,name='selu')
        elif activate=='swish':
            x = swish(x,name='swish')
        elif activate=='relu':
            x = tf.nn.relu(x,name='relu')
        elif activate=='relu6':
            x = tf.nn.relu6(x,name='relu6')
        elif activate=='prelu':
            x = prelu(x,name='prelu')
        else:
            pass

        return x
##weight variable
def GetWeight(name,shape,weights_decay = 0.00004):
    with tf.variable_scope(name):
        #w = tf.get_variable('weight',shape,tf.float32,initializer=VarianceScaling())
        w = tf.get_variable('weight',shape,tf.float32,initializer=glorot_uniform_initializer())
        #w = tf.get_variable('weight',shape,tf.float32,initializer=tf.random_normal_initializer(stddev=tf.sqrt(2.0/tf.to_float(shape[0]*shape[1]*shape[2]))))
        
        weight_decay = tf.multiply(tf.nn.l2_loss(w), weights_decay, name='weight_loss')
        tf.add_to_collection('regularzation_loss', weight_decay)
        return w
##initializer
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import math_ops
import math
def glorot_uniform_initializer(seed=None, dtype=dtypes.float32):
    return VarianceScaling(scale=1.0,
                          mode="fan_avg",
                          distribution="uniform",
                          seed=seed,
                          dtype=dtype)
def glorot_normal_initializer(seed=None, dtype=dtypes.float32):
    return VarianceScaling(scale=1.0,
                          mode="fan_avg",
                          distribution="normal",
                          seed=seed,
                          dtype=dtype)
def _compute_fans(shape):
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        receptive_field_size = 1.
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out
class VarianceScaling():
    def __init__(self, scale=1.0,
                 mode="fan_in",
                 distribution="normal",
                 seed=None,
                 dtype=dtypes.float32):
      if scale <= 0.:
          raise ValueError("`scale` must be positive float.")
      if mode not in {"fan_in", "fan_out", "fan_avg"}:
          raise ValueError("Invalid `mode` argument:", mode)
      distribution = distribution.lower()
      if distribution not in {"normal", "uniform"}:
          raise ValueError("Invalid `distribution` argument:", distribution)
      self.scale = scale
      self.mode = mode
      self.distribution = distribution
      self.seed = seed
      self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
      if dtype is None:
          dtype = self.dtype
      scale = self.scale
      scale_shape = shape
      if partition_info is not None:
          scale_shape = partition_info.full_shape
      fan_in, fan_out = _compute_fans(scale_shape)
      if self.mode == "fan_in":
          scale /= max(1., fan_in)
      elif self.mode == "fan_out":
          scale /= max(1., fan_out)
      else:
          scale /= max(1., (fan_in + fan_out) / 2.)
      if self.distribution == "normal":
          stddev = math.sqrt(scale)
          return random_ops.truncated_normal(shape, 0.0, stddev,
                                             dtype, seed=self.seed)
      else:
          limit = math.sqrt(3.0 * scale)
          return random_ops.random_uniform(shape, -limit, limit,
                                           dtype, seed=self.seed)
##batch_norm
def bn(x, is_training, name='batchnorm'):
    with tf.variable_scope(name):
        decay = 0.99
        epsilon = 1e-3
        
        size = x.shape.as_list()[-1]
        
        beta = tf.get_variable('beta', [size], initializer=tf.zeros_initializer())
        scale = tf.get_variable('scale', [size], initializer=tf.ones_initializer())

        moving_mean = tf.get_variable('mean', [size], initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = tf.get_variable('variance', [size], initializer=tf.ones_initializer(), trainable=False)

        def train():
            mean, variance = tf.nn.moments(x, [0,1,2])
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)
            return mean, variance

        mean, variance = tf.cond(
                                 tf.convert_to_tensor(is_training,dtype=tf.bool),
                                 lambda:train(),
                                 lambda:(moving_mean, moving_variance)
                                )
     
        inv = math_ops.rsqrt(variance + epsilon)
        inv *= scale 
        return x * inv + (beta - mean * inv)
##group_norm
def group_norm(x, eps=1e-5, name='group_norm'):
    with tf.variable_scope(name):
        _, _, _, C = x.get_shape().as_list()
        G = C//8
        
        #group_list = tf.split(tf.expand_dims(x,axis=3),num_or_size_splits=G,axis=4)#[(none,none,none,1,C//G),...]
        #x = tf.concat(group_list,axis=3)#none,none,none,G,C//G
        x = tf.reshape(x,tf.concat([tf.shape(x)[:3],tf.constant([G,C//G])],axis=0))#none,none,none,G,C//G
        
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)#none,none,none,G,C//G
        x = (x - mean) / tf.sqrt(var + eps)#none,none,none,G,C//G
        
        #group_list = tf.split(x,num_or_size_splits=G,axis=3)#[(none,none,none,1,C//G),...]
        #x = tf.squeeze(tf.concat(group_list,axis=4),axis=3)#none,none,none,C
        x = tf.reshape(x,tf.concat([tf.shape(x)[:3],tf.constant([C])],axis=0))#none,none,none,C

        gamma = tf.Variable(tf.ones([C]), name='gamma')
        beta = tf.Variable(tf.zeros([C]), name='beta')
        gamma = tf.reshape(gamma, [1, 1, 1, C])
        beta = tf.reshape(beta, [1, 1, 1, C])

    return x* gamma + beta
##LeakyRelu
def LeakyRelu(x, leak=0.1, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)
##selu
def selu(x,name='selu'):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)
##swish
def swish(x,name='swish'):
    with tf.variable_scope(name):
        beta = tf.Variable(1.0,trainable=True)
        return x*tf.nn.sigmoid(beta*x)
##crelu 
def crelu(x,name='crelu'):
    with tf.variable_scope(name):
        x = tf.concat([x,-x],axis=-1)
        return tf.nn.relu(x)
##prelu
def prelu(_x, name='prelu'):
    with tf.variable_scope(name):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1], dtype=_x.dtype, initializer=tf.constant_initializer(0.25))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

if __name__=='__main__':
    import time
    import numpy as np
    from functools import reduce
    from operator import mul

    def get_num_params():
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params

    model = AttentionResnet(num_classes=1000)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        num_params = get_num_params()
        print('all params:{}'.format(num_params))
        
        feed_dict = {model.input_image:np.random.randn(1,224,224,3),
                     model.y:np.array(1).reshape(-1,),
                     model.is_training:True
                    }
        
        start = time.time()
        out = sess.run(model.all_loss,feed_dict=feed_dict)
        print('Spend Time:{}'.format(time.time()-start))
        
        print(out)