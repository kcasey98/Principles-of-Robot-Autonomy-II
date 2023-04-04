import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import argparse
from utils import *

tf.config.run_functions_eagerly(True)

class NN(tf.keras.Model):
    def __init__(self, in_size, out_size):
        super(NN, self).__init__()
        
        ######### Your code starts here #########
        # We want to define and initialize the weights & biases of the neural network.
        # - in_size is dim(O)
        # - out_size is dim(A) = 2
        # IMPORTANT: out_size is still 2 in this case, because the action space is 2-dimensional. But your network will output some other size as it is outputing a distribution!
        # HINT: You should use either of the following for weight initialization:
        #         - tf.keras.initializers.GlorotUniform (this is what we tried)
        #         - tf.keras.initializers.GlorotNormal
        #         - tf.keras.initializers.he_uniform or tf.keras.initializers.he_normal
        
        # new sizes
        mean_size = out_size
        cov_size = out_size**2
        self.w1 = tf.keras.layers.Dense(in_size, activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.w2 = tf.keras.layers.Dense(64, activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.w3 = tf.keras.layers.Dense(64, activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.w4 = tf.keras.layers.Dense(64, activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.w5 = tf.keras.layers.Dense(64, activation='relu',kernel_initializer=tf.keras.initializers.GlorotUniform())
        # this makes the first two columns the mean vector and then next 4 columns the variances 
        self.w6 = tf.keras.layers.Dense(mean_size + cov_size) 
        
        ########## Your code ends here ##########

    def call(self, x):
        x = tf.cast(x, dtype=tf.float32)
        ######### Your code starts here #########
        # We want to perform a forward-pass of the network. Using the weights and biases, this function should give the network output for x where:
        # x is a (?, |O|) tensor that keeps a batch of observations
        # IMPORTANT: First two columns of the output tensor must correspond to the mean vector!
        
        # passing in the observations one layer at a time
        x = self.w1(x)
        x = self.w2(x)
        x = self.w3(x)
        x = self.w4(x)
        x = self.w5(x)
        x = self.w6(x)

        return x
        
        ########## Your code ends here ##########


   
def loss(y_est, y):
    y = tf.cast(y, dtype=tf.float32)
    ######### Your code starts here #########
    # We want to compute the negative log-likelihood loss between y_est and y where
    # - y_est is the output of the network for a batch of observations,
    # - y is the actions the expert took for the corresponding batch of observations
    # At the end your code should return the scalar loss value.
    # HINT: You may find the classes of tensorflow_probability.distributions (imported as tfd) useful.
    #       In particular, you can use MultivariateNormalFullCovariance or MultivariateNormalTriL, but they are not the only way.

    # Variables
    batch_size = y.shape[0] # batch size
    dim = 2 #dimension of matrix

    # find the mean
    mean = y_est[:,0:dim]

    # covariance matrices: n is batch_size 2 by 2
    cov = tf.reshape(y_est[:,dim:dim + dim**2],(batch_size,dim,dim))
    cov_t = tf.transpose(cov, [0, 2, 1])

    cov_final = cov@cov_t

    # This helps with errors
    delta = 1e-3
    id_delt = delta*tf.eye(2, batch_shape=[batch_size]) 

    cov_final += id_delt

    m = tfd.MultivariateNormalTriL(loc=mean, scale_tril=cov_final)
    l = -tf.reduce_mean(m.log_prob(y))
    return l
    
    ########## Your code ends here ##########

def nn(data, args):
    """
    Trains a feedforward NN. 
    """
    params = {
        'train_batch_size': 4096*32,
    }
    in_size = data['x_train'].shape[-1]
    out_size = data['y_train'].shape[-1]
    
    nn_model = NN(in_size, out_size)
    if args.restore:
        nn_model.load_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    print("here")

    @tf.function
    def train_step(x, y):
        ######### Your code starts here #########
        # We want to perform a single training step (for one batch):
        # 1. Make a forward pass through the model
        # 2. Calculate the loss for the output of the forward pass
        # 3. Based on the loss calculate the gradient for all weights
        # 4. Run an optimization step on the weights.
        # Helpful Functions: tf.GradientTape(), tf.GradientTape.gradient(), tf.keras.Optimizer.apply_gradients
       
        with tf.GradientTape() as t:
            l = loss(nn_model(x,training=True),y)
            w = nn_model.trainable_weights
            gradients = t.gradient(l,w)
            optimizer.apply_gradients(zip(gradients, w))


        ########## Your code ends here ##########

        train_loss(l)

    @tf.function
    def train(train_data):
        for x, y in train_data:
            train_step(x, y)


    train_data = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train'])).shuffle(100000).batch(params['train_batch_size'])

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()

        train(train_data)

        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1, train_loss.result()))

    # nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')

    print("save? [Y/n]")
    choice = input().lower()
    if choice in {"n", "no"}:
        print("NOT SAVING RESULTS")
    else:
        print('saving ./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')
        nn_model.save_weights('./policies/' + args.scenario.lower() + '_' + args.goal.lower() + '_ILDIST')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--goal', type=str, help="left, straight, right, inner, outer, all", default="all")
    parser.add_argument('--scenario', type=str, help="intersection, circularroad", default="intersection")
    parser.add_argument("--epochs", type=int, help="number of epochs for training", default=1000)
    parser.add_argument("--lr", type=float, help="learning rate for Adam optimizer", default=1e-3)
    parser.add_argument("--restore", action="store_true", default=False)
    args = parser.parse_args()
    
    maybe_makedirs("./policies")
    
    data = load_data(args)

    nn(data, args)
