#author:YIQING SHEN
#code of mixed-up self-distillation evalutation
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import math 
import matplotlib.pyplot as plt
import argparse
from scipy import integrate

print('keras version:',tf.keras.__version__)
print('tensorfflow version:',tf.__version__)

gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def read_data(data='Fashion_MNIST'):
    #print('data set:',data)
    mnist = tf.keras.datasets.cifar10 


    (data1,label1),(data2,label2) = mnist.load_data() 

    label1_vector = tf.keras.utils.to_categorical(label1)
    label2_vector = tf.keras.utils.to_categorical(label2)
    return data1,data2,label1_vector,label2_vector

def distribution_weight(a=1,b=2,mu=1,sigma=1):
    def gauss(x, mu, sigma):
        return math.exp(-((x-mu)**2)/(2*sigma**2))/(math.sqrt(math.pi*2)*sigma)
    value, error = integrate.quad(gauss, a, b, args = (mu, sigma))
    return value

def find_label_number(find_which_label, label): 
    count = 0
    label_list_form = np.argmax(label, axis=1)
    for i in range(label_list_form.size):
        if label_list_form[i]== find_which_label:
            count += 1
    return count

def find_label_index(find_which_label, label):
    label_list=[]
    label_list_form = np.argmax(label, axis=1)
    for i in range(label_list_form.size):
        if label_list_form[i]== find_which_label:
            label_list.append(i)
    return label_list

def relabel_dis(find_which_label,
               sigma,
               label):
    label_number = find_label_number (find_which_label = find_which_label, label = label ) 
    relabel_num =[]
    normalization_constant = 0
    for i in range(10):
        relabel_num.append(distribution_weight(i-0.5,i+0.5,mu=find_which_label,sigma = sigma))
        normalization_constant += relabel_num[i]
    for i in range(10):
        relabel_num[i]=int((label_number*relabel_num[i])/normalization_constant)
    delta = label_number - sum(relabel_num) 
    relabel_num[find_which_label] +=delta
    return relabel_num

def relabel(label,sigma):   
    relabel = np.argmax(label, axis=1)
    for find_which_label in range(10):
        relabel_num = relabel_dis(find_which_label = find_which_label,
                                 sigma = sigma,
                                 label = label)
        label_index = find_label_index(find_which_label = find_which_label, label = label)
        #print('label=',find_label,'relabel number =',relabel_num,label_index)
        for new_label in range(10):
            amount = relabel_num[new_label]
            if amount>0:
                for index in range(amount):
                    if len(label_index)>0:
                        ii = label_index.pop()
                        relabel[ii]=new_label
    relabel_vector = tf.keras.utils.to_categorical(relabel)
    return relabel_vector

def label_metrics(l1,l2):
    l1 = np.argmax(l1, axis=1)
    l2 = np.argmax(l2, axis=1)
    counter = 0
    for i in range(max(l1.shape)):
        if l1[i] == l2[i]:
            counter+=1 
    return counter /max(l1.shape)

def train_model(data,label,data_test,label_test,validation_data,validation_steps,batch_size,epochs):

    model = tf.keras.models.Sequential([
        #tf.keras.layers.Flatten(input_shape=(32,32,3)),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(2,2), strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation='relu',input_shape=(32,32,3)), 
        tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=1),      
        tf.keras.layers.Conv2D(filters=256,kernel_size=(2,2),strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation='relu'),
        #tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation ='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(512,activation ='relu'),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(512,activation ='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation ='softmax') ])

    model.compile(optimizer ='adam',#=tf.keras.optimizers.Adadelta(learning_rate=0.001), #'adam', 
                  loss=tf.keras.losses.CategoricalCrossentropy(),##'sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])#['accuracy'])

    history = model.fit(data,label,
        batch_size = batch_size,
        epochs = epochs,
        validation_data= validation_data,
        validation_steps=validation_steps)

    label_pred_v = model.predict(data,batch_size=batch_size)
    #label_pred=np.argmax(label_pred_v,axis=1)

    loss,accu = model.evaluate(data_test, label_test) 
    return label_pred_v,history,accu

def main(data='fashion_MNIST',iteration_num = 5,sigma = 0.0001, batch_size = 256,epochs = 10, lbd = 0.6):
    total_iterations = 0
    data1,data2,label1_vector,label2_vector=read_data(data)
    corrupt_label_vector = relabel (label=label1_vector,sigma=sigma)
    print('corrupt precentage %.4f'%(1-label_metrics (corrupt_label_vector,label1_vector)))
    inital_accu =label_metrics (corrupt_label_vector,label1_vector)
    training_accu = []
    test_accu = []
    test_list=[]
    for i in range(iteration_num):
        print('current iteration {} of all {} iterations'.format(i,iteration_num))
        epoch_this_iteration = epochs#int(epochs*(1+i/15))
        label_pred_vector,history,accu = train_model(data=data1,
                                                label=corrupt_label_vector,
                                                data_test=data2,
                                                label_test=label2_vector,
                                                validation_data=(data1,label1_vector),
                                                validation_steps=1,
                                                batch_size=batch_size,
                                                 epochs=epoch_this_iteration)
        #lbd1=lbd + (i/iteration_num)/10
        lbd1=lbd
        corrupt_label_vector = corrupt_label_vector*(1-lbd1) + label_pred_vector*lbd1
        corrupt_label_vector=tf.nn.softmax(corrupt_label_vector,axis=1)
        corrupt_label_vector=np.argmax(corrupt_label_vector,axis=1)
        corrupt_label_vector=tf.keras.utils.to_categorical(corrupt_label_vector)
        total_iterations += epoch_this_iteration
        test_list.append(total_iterations)
        # softmax layer
        #corrupt_label_vector = tf.nn.softmax(corrupt_label_vector,axis=1)

        training_accu += history.history['val_categorical_accuracy']
        test_accu.append(accu)
    corrupt_label_vector = relabel (label=label1_vector,sigma=sigma)
    label_pred_vector0,history0,accu0 = train_model(data=data1,
                                                label=corrupt_label_vector,
                                                data_test=data2,
                                                label_test=label2_vector,
                                                validation_data=(data1,label1_vector),
                                                validation_steps=1,
                                                batch_size=batch_size,
                                                 epochs=total_iterations)
    regular_training_accu=history0.history['val_categorical_accuracy']
    return training_accu,test_accu,regular_training_accu,inital_accu,total_iterations,test_list
    
if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--sigma", dest="S", type=float, default=0.3,
                        help="distribution")

    parser.add_argument("-i", "--iteration", dest="I", type=int, default=10,
                        help="Iteration times")

    parser.add_argument("-e", "--epoch", dest="E", type=int, default=10,
                        help="Epoches in each iteration")

    parser.add_argument("-b", "--batchsize", dest="B", type=int, default=512,
                        help="batch size")

    parser.add_argument("-d", "--dataset", dest="D", type=str, default='fashion_MNIST',
                        help="batch size")

    parser.add_argument("-l", "--lambda", dest="L", type=float, default=0.5,
                        help="batch size")


    args = parser.parse_args()
    data = args.D
    sigma = args.S
    iteration_num = args.I
    epochs= args.E
    batch_size = args.B
    lbd= args.L

    training_accu,test_accu,regular_training_accu,inital_accu,total_iterations,test_list = main(data= data ,
        iteration_num = iteration_num,sigma = sigma, batch_size = batch_size,epochs = epochs, lbd = lbd)
    xlist=list(range(epochs,total_iterations+1,epochs))
    maxc = int((total_iterations+1)/10)
    xlist_show = list(range(epochs,total_iterations+1,maxc))
    xlist_hi=list(range(1,total_iterations+1,1))
    fig=plt.figure()
    fig.set_size_inches(10,8)
    plt.title('Learning behaviour of DNN \n(initial label accuracy %.4f)'% inital_accu)
    plt.plot(xlist_hi,training_accu,'b-',label='accu. on train set',alpha=0.5)
    plt.plot(test_list,test_accu,'g-',label='accu. on test set')
    plt.plot(xlist_hi,regular_training_accu,color='gold',linestyle='-',alpha=0.5)
    plt.legend(['train accu.','test accu.','regular train accu.'],loc='lower left')##upper right
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.axis([1,total_iterations+1,0.0,1.0])
    ax = plt.gca()
    ax.set_xticks(xlist_show)
    fig.savefig('cifar10 result(batch{},iterations{},epochs{},lambda{}).png'.format(batch_size,iteration_num,epochs,lbd),dpi=200)



