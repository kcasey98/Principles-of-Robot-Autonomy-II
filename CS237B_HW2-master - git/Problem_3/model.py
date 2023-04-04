import tensorflow as tf 

DIM_IMG = (224, 224)

class AccelerationLaw(tf.keras.layers.Layer):
    """
    Tensorflow layer to evaluate the acceleration law:

        a = g * (sin(th) - mu * cos(th))

    g is a trainable parameter because the units of acceleration in the
    dataset are pixels/frame^2, and the conversion from 9.81 m/s^2 to these
    units are unknown.
    """

    def __init__(self, **kwargs):
        super(AccelerationLaw, self).__init__(**kwargs)

    def build(self, input_shape):
        self.g = self.add_weight(name='g', shape=(1,), initializer=tf.keras.initializers.Constant(16), trainable=True)

    def call(self, inputs):
        mu, th = inputs

        ########## Your code starts here ##########
        a = self.g * (tf.math.sin(th) - mu * tf.math.cos(th))  # TODO
        ########## Your code ends here ##########

        # Ensure output acceleration is positive
        return a

def build_model():
    """
    Build the acceleration prediction network.

    The network takes two inputs:
        img - first frame of the video
        th  - incline angle of the ramp [rad]

    The output is:
        a - predicted acceleration of the object [pixels/frame^2]

    The last two layers of the network before the AccelerationLaw layer should be:
        p_class - A fully connected layer of size 32 with softmax output. This
                  represents a probability distribution over 32 possible classes
                  for the material of the object.
                  NOTE: Name this layer 'p_class'!
        mu - A vector of 32 weights representing the friction coefficients of
             each material class. The dot product of these weights and p_class
             represent the predicted friction coefficient of the object in the
             video.
             NOTE: Name this layer 'mu'!
    """

    img_input = tf.keras.Input(shape=(DIM_IMG[1], DIM_IMG[0], 3), name='img')
    th_input = tf.keras.Input(shape=(1,), name='th')

    ########## Your code starts here ##########
    # TODO: Create your neural network and replace the following two layers
    #       according to the given specification.

    # First Layer: Convolution, Batch Norm and Activation
    c1 = tf.keras.layers.Conv2D(32, (3,3))(img_input)
    b1 = tf.keras.layers.BatchNormalization()(c1)
    a1 = tf.keras.layers.Activation('relu')(b1)

    # then Dropout and Pool
    d1 = tf.keras.layers.Dropout(0.5)(a1)
    m1 = tf.keras.layers.MaxPooling2D((2,2))(d1)

    # Second Layer: Convolution, Batch Norm and Activation
    c2 = tf.keras.layers.Conv2D(32, (3,3))(m1)
    b2 = tf.keras.layers.BatchNormalization()(c2)
    a2 = tf.keras.layers.Activation('relu')(b2)

    # then Dropout and Pool
    d2 = tf.keras.layers.Dropout(0.3)(a2)
    m2 = tf.keras.layers.MaxPooling2D((2,2))(d2)

    fl = tf.keras.layers.Flatten()(m2)

    p_class = tf.keras.layers.Dense(32, name='p_class',activation = 'softmax')(fl)
    mu = tf.keras.layers.Dense(32, name='mu')(p_class)
    # This is the sum of p times coefficient of friction
    pmu = tf.math.multiply(p_class,mu)
    mu = tf.math.reduce_sum(pmu,axis = -1,keepdims=True)

    ########## Your code ends here ##########

    a_pred = AccelerationLaw(name='a')((mu, th_input))

    return tf.keras.Model(inputs=[img_input, th_input], outputs=[a_pred])

def build_baseline_model():
    """
    Build a baseline acceleration prediction network.

    The network takes one input:
        img - first frame of the video

    The output is:
        a - predicted acceleration of the object [pixels/frame^2]

    The structure of this network should match the other model before the
    p_class layer. Instead of outputting p_class, it should directly output a
    scalar value representing the predicted acceleration (without using the
    AccelerationLaw layer).
    """

    img_input = tf.keras.Input(shape=(DIM_IMG[1], DIM_IMG[0], 3), name='img')
    th_input = tf.keras.Input(shape=(1,), name='th')

    ########## Your code starts here ##########
    # TODO: Replace the following with your model from build_model().

    # First Layer: Convolution, Batch Norm and Activation
    c1 = tf.keras.layers.Conv2D(32, (3,3))(img_input)
    b1 = tf.keras.layers.BatchNormalization()(c1)
    a1 = tf.keras.layers.Activation('relu')(b1)

    # then Dropout and Pool
    d1 = tf.keras.layers.Dropout(0.5)(a1)
    m1 = tf.keras.layers.MaxPooling2D((2,2))(d1)

    # Second Layer: Convolution, Batch Norm and Activation
    c2 = tf.keras.layers.Conv2D(32, (3,3))(m1)
    b2 = tf.keras.layers.BatchNormalization()(c2)
    a2 = tf.keras.layers.Activation('relu')(b2)

    # then Dropout and Pool
    d2 = tf.keras.layers.Dropout(0.3)(a2)
    m2 = tf.keras.layers.MaxPooling2D((2,2))(d2)

    fl = tf.keras.layers.Flatten()(m2)

    a_pred = tf.keras.layers.Dense(1,name = 'a')(fl)
    ########## Your code ends here ##########

    return tf.keras.Model(inputs=[img_input, th_input], outputs=[a_pred])

def loss(a_actual, a_pred):
    """
    Loss function: L2 norm of the error between a_actual and a_pred.
    """

    ########## Your code starts here ##########
    l = tf.norm(a_actual - a_pred, ord=2,axis=1)  # TODO
    ########## Your code ends here ##########

    return l
