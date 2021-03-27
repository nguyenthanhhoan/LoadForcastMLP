import tensorflow as tf
import numpy as np
import copy, datetime, statistics

class MLP():

    def __init__(self,n_inputs,n_hidden,n_outputs,learning_rate=0.001,batch_size=50,training_epoch=50):
        """
        Khởi tạo hyperparameter và MLP cùng với các thông số khác như hàm mất mát, phương pháp thay đổi trọng số
        """
        self.n_inputs = n_inputs #Layer input
        self.n_hidden = n_hidden # Layer ẩn 1
        self.n_outputs = n_outputs #Layer output
        self.learning_rate = learning_rate #Tốc độ học
        self.batch_size = batch_size #Số lượng dòng data cho mỗi lần học
        self.training_epoch = training_epoch #Số  lần train lại cho MLP
        #Khởi tạo layer input và "layer output chứa kết quả thật"
        self.X = tf.placeholder("float", [None,n_inputs])
        #self.X = tf.Variable(tf.ones(shape=[None, n_inputs]), dtype=tf.float32)
        self.Y = tf.placeholder("float", [None,n_outputs])
        #self.Y = tf.Variable(tf.ones(shape=[None, n_outputs]), dtype=tf.float32)

        #weight and bias cho các node trong MLP
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_inputs, n_hidden])),
            'out': tf.Variable(tf.random_normal([n_hidden, n_outputs]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden])),
            'out': tf.Variable(tf.random_normal([n_outputs]))
        }

        #Layer output chứa kết quả dự đoán từ MLP
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
        weights = tf.trainable_variables()
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        self.YPred = self.__multilayer_perceptron(self.X) #Xem dòng 41 trở đi
        self.loss_op = tf.losses.absolute_difference(self.Y,self.YPred) + regularization_penalty #Thêm hàm error, trong trường hợp này là absolute difference
        self.optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=self.learning_rate) #Hàm thay đổi trọng số Adam Optimizer (cải tiến của Stochastic GD). 
        self.train_op = self.optimizer.minimize(self.loss_op) #Phương pháp train: Giảm thiểu error

    def __multilayer_perceptron(self,X):
        """
        Định nghĩa MLP thực hiện như thế nào
        """
        layer_1 = tf.add(tf.matmul(X, self.weights['h1']), self.biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.nn.relu(layer_1),tf.sin(layer_1)) #sin function is similar to any kind of wave functions
        #layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        #layer_2 = tf.sin(layer_2) #sin function is similar to any kind of wave functions -> is it possible to use cos function?
        #layer_3 = tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3'])
        #layer_3 = tf.sin(layer_3)
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_1, self.weights['out']) + self.biases['out']
        out_layer = tf.nn.relu(out_layer) #gettting positive value; omit any other values
        return out_layer

    def train_save_and_log(self,X_data,Y_data,fold=5,save_directory=None,model_name=None):
        logger = open(save_directory + "/log.txt",'a')
        """
        Train dữ liệu dùng forward chaining
        """
        #Khởi tạo
        self.init = tf.global_variables_initializer()
        #Chia dữ liệu thành fold
        fold_size = int(len(X_data) / fold)
        #Bắt đầu  MLP
        logger.write("Begin training on {}\n".format(datetime.datetime.now()))
        with tf.Session() as sess:
            sess.run(self.init)
            k_score = []
            for i in range(0,fold - 1):
                #1. PARSING THE DATA
                X_test = X_data[fold_size * (i + 1) : fold_size * (i + 2)]
                Y_test = Y_data[fold_size * (i + 1) : fold_size * (i + 2)]
                X_train = X_data[:fold_size * (i + 1)]
                Y_train = Y_data[:fold_size * (i + 1)]
                #########################################################
                #2. TRAINING THE DATA
                epoch_error = 0
                for epoch in range(self.training_epoch): #for each epoch
                    total_batch = int(len(X_test) / self.batch_size)
                    avg_error = 0
                    for batch in range(total_batch): #for each batch in X_train and Y_train
                        batch_x = np.asarray(X_train[batch * self.batch_size : (batch + 1) * self.batch_size])
                        batch_y = np.asarray(Y_train[batch * self.batch_size : (batch + 1) * self.batch_size])
                        _,err = sess.run([self.train_op,self.loss_op],feed_dict={self.X: batch_x,self.Y: batch_y})
                        avg_error += err/total_batch #cost cho tổng input 

                    if epoch % 3 == 0:
                        logger.write('fold: {}, epoch: {}, mean_error = {}\n'.format(i + 1,epoch + 1,avg_error))
                    epoch_error += avg_error / self.training_epoch
                #########################################################
                #2. EVALUATE DATA WITH R2 SCORE FUNCTION
                pred = self.YPred.eval(feed_dict={self.X:X_test})
                savepred = numpy.asarray(pred)
                numpy.savetxt("predict.csv", savepred, delimiter=",")
                Y_pred = [x[0] for x in pred]
                Y_test = [x[0] for x in Y_test]
                g = 0
                for i in range(len(Y_test)):
                    x = (Y_test[i] - Y_pred[i]) ** 2
                    g = g + x
                m = statistics.mean(Y_test)
                my = 0
                for i in range(len(Y_test)):
                    my = my + ((Y_test[i] - m) ** 2)
                s = 1 - (g / my)
                k_score = k_score + [s]
            print("Final error list: ")
            print(k_score)
            logger.write('Result: {}' + str(k_score))
            saver = tf.train.Saver()
            save_path = saver.save(sess,model_name)
            print("Model saved in {}".format(save_path))
            logger.close()
