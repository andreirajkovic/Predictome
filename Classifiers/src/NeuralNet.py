from __future__ import division, print_function, unicode_literals
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import dropout

class Graph(object):
    def __init__(self, trainX, trainy, parameters={'layers': 2, 'neurons': [10, 10], 'penalty': 'dropout', 'keep': 0.5, 'C': 0.05, 'lin:reg': False}):
        """
        Setup neural network from TensorFlow
        www.tensorflow.org
        
        Parameters:
        ======================================
        
        trainX -> A 2D numpy array of the explainatory variables cannot be sparse
        
        trainy -> Numpy array of the response variables

        parameters -> Dictionary of parameters


        Attributes:
        ======================================

        graph -> TensorFlow Graph

        rate -> Float as a placeholder for the dropout rate

        n_output -> Integer: Sets the number of outputs in the final layer

        X -> Tensor Placeholder: The matrix to hold the training data

        y -> Tensor Placeholder: The vector to hold the repsonse variable

        is_training -> Tensor Placeholder boolean: Holds the status of training or not

        loss-> Tensor Variable: Refers to the loss computations

        accuracy -> Tensor Variable: Refers to the accuracy computations
        
        r2 -> Tensor Variable: Refers to the computations of an r-squared value

        probs -> Tensor Variable: Refers to the computations of the probabilties 
                 for a repsonse variable

        logit -> Tensor Variable: Refers to the computations for the raw logit output

        optimizer -> Tensor Variable: Refers to the computations for the Adam optimization

        training_op -> Tensor Variable: Refers to the computations to minimize the loss

        init -> Tensor Variable: Initiates the global variables in the tensor graph

        n_epochs -> Positive integer: Number of epochs set to 50 as default

        batchsize -> Positive integer: Split of the data into batches of 50

        train_size -> Positive integer: Number of training samples


        Notes:
        ======================================

        None

        Examples:
        ======================================
        
        None
        """

        # graph parameters
        self.graph = tf.Graph()
        self.rate = 1.0
        with self.graph.as_default():
            self.trainX = trainX
            self.trainy = trainy
            if parameters['lin:reg'] is False:
                self.n_output = len(np.unique(trainy))
            else:
                self.n_output = 1
            self.parameters = parameters
            # fully connected parameters
            if parameters['penalty'] == 'dropout':
                self.rate = parameters['keep']
            elif parameters['penalty'] == 'l1' or 'l2':
                self.C = parameters['C']
            # architecture parameters
            self.X = tf.placeholder(dtype=tf.float32, shape=(
                                    None, trainX.shape[1]), name="X")
            if parameters['lin:reg'] is False:
                self.y = tf.placeholder(dtype=tf.int64, shape=(None), name="y")
            else:
                self.y = tf.placeholder(dtype=tf.float32, shape=(None), name="y")
            self.is_training = tf.placeholder(
                tf.bool, shape=(), name='is_training')
            self.loss = 0
            self.accuracy = 0
            self.r2 = 0
            self.probs = 0
            self.logit = 0
            self.optimizer = 0
            self.training_op = 0
            self.architecture()
            """ ALL TF VARIABLES BEFORE """
            self.init = tf.global_variables_initializer()
            # model parameters
            self.n_epochs = 50
            self.batchsize = 50
            self.train_size = self.trainX.shape[0]

    def fit_predict(self, valX, valy):
        """ Model is trained and then runs a prediction at the end"""
        probs = self.runModel(valX=valX, valy=valy)
        return probs

    def reset_graph(self):
        """Reset a graph and all the nodes"""        
        tf.reset_default_graph()
        np.random.seed()

    def fetch_batch(self, batchsize, x_train, y_train):
        """Randomly select training instances for the batch without replacement"""        
        indices = np.random.choice(np.arange(len(x_train)), size=batchsize, replace=False)
        x_train_batch = x_train[indices]
        y_train_batch = y_train[indices]
        return x_train_batch, y_train_batch

    def classBalance(self, x_train, y_train, y_size):
        """Balances the number of instances if the problem is classification"""        
        trainShuffled = np.random.choice(
            y_size, size=y_size, replace=False).astype(int)
        x_bala = np.zeros([y_size, x_train.shape[1], x_train.shape[2], 1])
        y_bala = np.zeros(y_size)
        label_size = y_size // np.unique(y_train).shape[0]
        i = 0
        for label in np.unique(y_train):
            label_indx = np.where(y_train == label)
            random_indx = np.random.choice(
                label_indx[0], size=label_size, replace=False)
            x_bala[i:label_size + i] = x_train[random_indx]
            y_bala[i:label_size + i] = y_train[random_indx]
            i += label_size
        x_bala_shuf = x_bala[trainShuffled]
        y_bala_shuf = y_bala[trainShuffled]
        return (x_bala_shuf, y_bala_shuf)

    def dense_dropout(self, inputs, rate, neurons, is_training):
        """makes a dense dropout layer of neurons"""        
        dropout_layer = dropout(inputs, rate, is_training=is_training)
        dense_layer = tf.layers.dense(dropout_layer, neurons,
                                      activation=tf.nn.relu)
        return dense_layer

    def architecture(self):
        """
        Sets up a series of layers and neurons with ReLU functions
        Can perform both regression and classification
        """        
        layers = []
        layers.append(self.dense_dropout(self.X, self.rate, self.parameters['neurons'][0], is_training=self.is_training))
        for l in range(self.parameters['layers']):
            if l + 1 == self.parameters['layers']:
                break
            layers.append(self.dense_dropout(layers[l], self.rate, self.parameters['neurons'][l + 1], is_training=self.is_training))
        layers.append(tf.layers.dense(layers[self.parameters['layers'] - 1], self.n_output, activation=None, name="logits"))
        self.logit = layers[-1]
        if self.parameters['lin:reg'] is False:
            # CLASSIFICATION
            T = tf.one_hot(self.y, depth=self.n_output)
            xentropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=layers[-1], labels=T)
            self.loss = tf.reduce_mean(xentropy)
            # Predictions
            with tf.name_scope("SoftMaxPrediction"):
                self.probs = tf.nn.softmax(layers[-1])
                y_proba_argmax = tf.argmax(self.probs, axis=1)
                y_pred = tf.squeeze(y_proba_argmax)
            with tf.name_scope("Accuracy"):
                correct = tf.equal(self.y, y_pred)
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct, tf.float32), name="accuracy")
        else:
            # REGRESSION
            mse = tf.losses.mean_squared_error(labels=self.y, predictions=self.logit)
            self.loss = tf.reduce_mean(mse)
            # Predictions
            with tf.name_scope("R2"):
                residual = tf.reduce_sum(tf.square(tf.subtract(self.y, self.logit)))
                total = tf.reduce_sum(tf.square(tf.subtract(self.y, tf.reduce_mean(self.y))))
                self.accuracy = tf.subtract(1.0, tf.div(residual, total), name="r2")

        # TRAINING LOSS
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.training_op = self.optimizer.minimize(self.loss)

    def runModel(self, valX=None, valy=None):
        """
        Runs the model for 50 epochs iterating over the data in batches of 50.
        After the model has been trained a prediction is executed and the
        logits and probability, or r2 is returned to the score_model fucntion. 
        """
        with tf.Session(graph=self.graph) as sess:
            self.init.run(session=sess)
            for epoch in range(self.n_epochs):
                n_batches = int(np.ceil(self.train_size / self.batchsize))
                for iteration in range(1, n_batches + 1, 1):
                    x_train_b, y_train_b = self.fetch_batch(self.batchsize,
                                                            self.trainX, self.trainy)
                    loss_train, _, train_error_rate = sess.run([self.loss,
                                                                self.training_op,
                                                                self.accuracy],
                                                               feed_dict={
                                                               self.X: x_train_b,
                                                               self.y: y_train_b,
                                                               self.is_training: True})

                    # if iteration % 100 == 0:
                    #     print("y")
                    #     weight = tf.get_default_graph().get_tensor_by_name('dense_1/kernel:0')
                    #     w = weight.eval(feed_dict={self.X: x_train_b,
                    #                                 self.y: y_train_b,
                    #                                 self.is_training: True})
                    #     plt.boxplot(w)
                    #     plt.show()

                    print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.3f} Epoch: {:1d} Acc: {:.1f} ".format(iteration, n_batches, iteration * 100 / n_batches, loss_train, epoch, train_error_rate), end="\r")

            if self.parameters['lin:reg'] is False:
                prob = self.probs.eval(session=sess, feed_dict={self.X: valX,
                                       self.y: valy,
                                       self.is_training: False}
                                       )

                logits = self.logit.eval(session=sess, feed_dict={self.X: valX,
                                         self.y: valy,
                                         self.is_training: False})
                return prob, logits
            else:
                logits = self.logit.eval(session=sess, feed_dict={self.X: valX,
                                         self.y: valy,
                                         self.is_training: False})
                r2 = self.accuracy.eval(session=sess, feed_dict={self.X: valX,
                                        self.y: valy,
                                        self.is_training: False})
                return r2, logits
