import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard, Callback


def concat_features(filename, features, rows):
    cur=pd.read_csv(filename, nrows=rows)
    join=pd.merge(cur, features, left_on='img_id_A', right_on='img_id')
    res=(pd.merge(join, features, left_on='img_id_B', right_on='img_id'))
    res=res.drop(['img_id_x','img_id_B', 'img_id_A', 'img_id_y'], axis=1)
    return res.loc[:, ~res.columns.str.contains('^Unnamed')]

def sub_features(df, num):
    new_df=copy.deepcopy(df)
    for i in range(1, num+1):
        new_df['new_f'+str(i)] = (new_df['f'+str(i)+'_x'] - new_df['f'+str(i)+'_y']).abs()
    return new_df.loc[:, ~new_df.columns.str.contains('^f')]

def read_csv(filename):
    df=pd.read_csv(filename)
    return df


def split_data(data,target):
    train=data.sample(frac=0.8,random_state=200)
    train_target=target.iloc[train.index]
    validation=data.drop(train.index).sample(frac=0.5,random_state=200)
    validation_target=target.iloc[validation.index]
    test=data.drop(train.index).drop(validation.index)
    test_target=target.iloc[test.index]
    return(train, train_target.values, validation, validation_target.values, test, test_target.values)

def get_phi(data, big_sigma, mu):
    phi = np.zeros((len(data.index), len(mu)))
    big_sigma_inv = np.linalg.inv(big_sigma)
    for i, x in enumerate(data.values):
        for j, m in enumerate(mu):
            tem = math.exp(-(1/2)*np.dot(np.dot(np.transpose(np.subtract(x,m)),big_sigma_inv),np.subtract(x,m)))
            phi[i][j]=tem
    return phi

def get_big_sigma(training_feature, mu):
    big_sigma = np.zeros((len(training_feature.columns),len(training_feature.columns)))
    selector = VarianceThreshold()
    selector.fit_transform(training_feature.values)
    for i, variances in enumerate(selector.variances_):
        big_sigma[i][i]=variances*300
    return big_sigma

def log_loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def find_erms(test_out, target, to_list=True):
    square_sum=0
    if to_list:
        test_out=list(np.concatenate(test_out))
        target=list(np.concatenate(target))
    count=0
    for i in range(len(test_out)):
        square_sum+=(test_out[i]-target[i])**2
        if(int(round(test_out[i]))==target[i]):
            count+=1
    return(math.sqrt(square_sum/len(test_out)), count/len(test_out))

drop_out = 0.20
first_dense_layer_nodes  = 256
second_dense_layer_nodes = 256
third_dense_layer_nodes = 2
def get_model(input_size):

    model = Sequential()
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))

    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))

    model.add(Dense(third_dense_layer_nodes))
    model.add(Activation('softmax'))
    model.summary()

    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

humanobserved_files=['HumanObserved-Dataset/HumanObserved-Features-Data/HumanObserved-Features-Data.csv',
                    'HumanObserved-Dataset/HumanObserved-Features-Data/same_pairs.csv',
                    'HumanObserved-Dataset/HumanObserved-Features-Data/diffn_pairs.csv']
h_row=791
gsc_files=['GSC-Features.csv', 'same_pairs.csv', 'diffn_pairs.csv']
gcs_rows=7000
for ff, rr, cc in zip([humanobserved_files, gsc_files],[h_row, gcs_rows], [9, 512]):
    rows=rr
    files=ff

    all_image_features=read_csv(files[0])
    con_same_pair=concat_features(files[1], all_image_features, rows)
    con_diff_pair=concat_features(files[2], all_image_features, rows)

    con = pd.concat( [con_same_pair, con_diff_pair], axis=0, ignore_index=True)
    sub=sub_features(con, cc)

    con=con.loc[:, (con != con.iloc[0]).any()]
    sub=sub.loc[:, (sub != sub.iloc[0]).any()]
    cluster=15

    for flag, method in enumerate([sub, con]):
        training_feature, train_target, validation_feature, validation_target, testing_feature, testing_target=split_data(method.loc[:, method.columns != 'target'], method.iloc[:,0])

        ###################### Linear Regression ###############################
        start_w=np.zeros(cluster)
        kmeans = KMeans(n_clusters=cluster, random_state=0).fit(training_feature.values)
        mu = kmeans.cluster_centers_
        big_sigma = get_big_sigma(training_feature, mu)
        print("training phi...")
        print("shape: ", training_feature.shape)
        t_phi = get_phi(training_feature, big_sigma, mu)

        print("validation phi...")
        print("shape: ", validation_feature.shape)
        val_phi = get_phi(validation_feature, big_sigma, mu)

        print("testing phi...")
        print("shape: ", testing_feature.shape)
        test_phi = get_phi(testing_feature, big_sigma, mu)


        e_lambda=2
        lr=0.001
        all_train_erm, all_test_erm, all_val_erm=[], [], []
        all_train_acc, all_test_acc, all_val_acc=[], [], []
        for i in range(rows):
            del_e = -np.dot((train_target[i]-np.dot(np.transpose(start_w), t_phi[i])), t_phi[i])
            la_del=np.dot(e_lambda,start_w)
            del_e = np.add(del_e,la_del)
            del_w= -np.dot(lr, del_e)
            w_next=del_w+start_w
            start_w=w_next

            train_out=np.dot(np.transpose(w_next), np.transpose(t_phi))
            train_erm, train_acc = find_erms(train_out, train_target, False)
            all_train_erm.append(train_erm)
            all_train_acc.append(train_acc)

            val_out = np.dot(np.transpose(w_next), np.transpose(val_phi))
            val_erm, val_acc = find_erms(val_out, validation_target, False)
            all_val_erm.append(val_erm)
            all_val_acc.append(val_acc)

            test_output = np.dot(np.transpose(w_next), np.transpose(test_phi))
            test_erm, test_acc = find_erms(test_output, testing_target, False)
            all_test_erm.append(test_erm)
            all_test_acc.append(test_acc)


        plt.plot(range(rows), all_val_erm, label = 'Validation')
        plt.plot(range(rows), all_train_erm, label = 'Training')
        plt.plot(range(rows), all_test_erm, label = 'Test')

        plt.legend()
        plt.ylabel('Erms')
        plt.show()



        ###################### Neural Networks ###############################

        if flag == 0:
            input_size=sub.shape[1]-1
        else:
            input_size=con.shape[1]-1


        num_epochs = 700
        model_batch_size = 128
        tb_batch_size = 32
        early_patience = 100
        model = get_model(input_size)
        tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
        earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=0, patience=early_patience, mode='min')
        #model fitting
        history = model.fit(training_feature.values
                            , np_utils.to_categorical(np.array(train_target), 2)
                            , verbose=0
                            , validation_split=0.2
                            , epochs=num_epochs
                            , batch_size=model_batch_size
                            , callbacks = [tensorboard_cb, earlystopping_cb]) # can add plot_losses callback

        training_loss = history.history['loss']
        test_loss = history.history['val_loss']
        training_acc = history.history['acc']
        test_acc = history.history['val_acc']
            #Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

            # Visualize loss history
        plt.plot(epoch_count, training_loss)
        plt.plot(epoch_count, test_loss)
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show();
            # Visualize Acc history
        plt.plot(epoch_count, training_acc)
        plt.plot(epoch_count, test_acc)
        plt.legend(['Training Acc', 'Test Acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.show();

        wrong   = 0
        right   = 0

        for i,j in zip(testing_feature.values, np_utils.to_categorical(np.array(testing_target), 2)):
            y = model.predict(np.array(i.tolist()).reshape(-1,input_size))
            if j.argmax() == y.argmax():
                right = right + 1
            else:
                wrong = wrong + 1

        print("Errors: " + str(wrong), " Correct :" + str(right))
        print("Testing Accuracy: " + str(right/(right+wrong)*100))




        ###################### Logistic Regression ###############################

        lr=0.001
        theta=np.zeros(training_feature.shape[1])
        all_train_loss, all_test_loss, all_val_loss=[], [], []
        all_train_acc, all_test_acc, all_val_acc=[], [], []
        for i in range(10000):
            z = np.dot(training_feature, theta)
            h = sigmoid(z)
            gradient = np.dot(training_feature.T, (h - training_target)) / train_target.size
            theta -= lr * gradient
            train_loss = log_loss(h, training_target)

            count=0
            for i in range(len(h)):
                if(train_target[i]==round(h[i])):
                    count+=1
            all_train_acc.append(count/len(h))
           # print(count/len(h))
            #print(f'training loss: {train_loss} \t')
            all_train_loss.append(train_loss)

            z = np.dot(validation_feature, theta)
            h = sigmoid(z)
            val_loss=log_loss(h, validation_target)
            #print(f'validation loss: {train_loss} \t')
            count=0
            for i in range(len(h)):
                if(validation_target[i]==round(h[i])):
                    count+=1
           # print(count/len(h))
            all_val_acc.append(count/len(h))
            all_val_loss.append(val_loss)

            z = np.dot(testing_feature, theta)
            h = sigmoid(z)
            test_loss=log_loss(h, testing_target)
            count=0
            for i in range(len(h)):
                if(testing_target[i]==round(h[i])):
                    count+=1
           # print(count/len(h))
            all_test_acc.append(count/len(h))
            #print(f'testing loss: {test_loss} \t')
            all_test_loss.append(test_loss)
        plt.plot(range(10000), all_val_loss, label = 'Validation')
        plt.plot(range(10000), all_train_loss, label = 'Training')
        plt.plot(range(10000), all_test_loss, label = 'Test')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.plot(range(10000), all_val_acc, label = 'Validation')
        plt.plot(range(10000), all_train_acc, label = 'Training')
        plt.plot(range(10000), all_test_acc, label = 'Test')
        plt.ylabel('Acc')
        plt.legend()
        plt.show()
