from __future__ import print_function
import inbreast
import keras.backend as K
from roc_auc import RocAucScoreOp, PrecisionOp, RecallOp, F1Op
from roc_auc import AUCEpoch, PrecisionEpoch, RecallEpoch, F1Epoch, LossEpoch, ACCEpoch
#from keras.preprocessing.image import ImageDataGenerator
from image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, SpatialDropout2D
from keras.layers import advanced_activations
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1l2
import inbreast
#import googlenet
from convnetskeras.convnets import preprocess_image_batch, convnet
# 基于稀疏深度MIL
np.random.seed(1)
#srng = RandomStreams(1)
fold = 4# 4
valfold = 0
lr = 5e-5
nb_epoch = 500
batch_size = 80
l2factor = 5e-6
l1factor = 0#2e-7
usedream = False # 是否使用某种特定方法
weighted = False #True
noises = 50
data_augmentation = True
modelname = 'alexnet' # miccai16, alexnet, levynet, googlenet
pretrain = True
#sparsemil = True
sparsemil = True 是否使用稀疏多实例学习
sparsemill1 = 1e-4 #1e-5 # 稀疏MIL的L1正则化系数
sparsemill2 = 0.0 #1e-2 # 稀疏MIL的L2正则化系数
savename = modelname+'60new_fd'+str(fold)+'_vf'+str(valfold)+'_lr'+str(lr)+'_l2'+str(l2factor)+'_l1'\
+str(l1factor)+'_ep'+str(nb_epoch)+'_bs'+str(batch_size)+'_w'+str(weighted)+'_dr'+str(usedream)+str(noises)+str(pretrain)+'_sp'+str(sparsemil)+str(sparsemill1)+str(sparsemill2)+'ft'
print(savename)
nb_classes = 2
# input image dimensions
img_rows, img_cols = 227, 227
# the CIFAR10 images are RGB
img_channels = 1

# the data, shuffled and split between train and test sets
# 从 inbreast 模块加载数据，包括训练集、验证集和测试集
trX, y_train, teX, y_test, teteX, y_test_test = inbreast.loaddataenhance(fold, 5, valfold=valfold, valnum=60)
# 将标签数据重塑为二维数组
trY = y_train.reshape((y_train.shape[0],1))
teY = y_test.reshape((y_test.shape[0],1))
teteY = y_test_test.reshape((y_test_test.shape[0],1))
''' for finetune '''
# 将验证集数据合并到训练集中，用于微调 finetune
trX = np.concatenate((trX, teX), axis=0)
y_train = np.concatenate((y_train,y_test), axis=0)
teX = teteX
y_test = y_test_test
print('tr, val, te pos num and shape')
print(trY.sum(), teY.sum(), teteY.sum(), trY.shape[0], teY.shape[0], teteY.shape[0])
ratio = trY.sum()*1./trY.shape[0]*1.
print('tr ratio'+str(ratio))
weights = np.array((ratio, 1-ratio))
#trYori = np.concatenate((1-trY, trY), axis=1)
#teY = np.concatenate((1-teY, teY), axis=1)
#teteY = np.concatenate((1-teteY, teteY), axis=1)
# 将输入数据重塑为适合卷积神经网络的形状（通道在前格式）
X_train = trX.reshape(-1, img_channels, img_rows, img_cols)
X_test = teX.reshape(-1, img_channels, img_rows, img_cols)
X_test_test = teteX.reshape(-1, img_channels, img_rows, img_cols)
print('tr, val, te mean, std')
print(X_train.mean(), X_test.mean(), X_test_test.mean())
# convert class vectors to binary class matrices
#Y_train = np.zeros((y_train.shape[0], mysoftmilbigk+mysoftmilsmallk))
#Y_test = np.zeros((y_test.shape[0], mysoftmilbigk+mysoftmilsmallk))
#Y_test_test = np.zeros((y_test_test.shape[0], mysoftmilbigk+mysoftmilsmallk))
#for i in xrange(mysoftmilsmallk,mysoftmilbigk+mysoftmilsmallk):
#  #Y_train[:,i,0] = 1-y_train
#  Y_train[:,i] = y_train
#  #Y_test[:,i,0] = 1-y_test
#  Y_test[:,i] = y_test
#  #Y_test_test[:,i,0] = 1-y_test_test
#  Y_test_test[:,i] = y_test_test
#for i in xrange(mysoftmilsmallk):
#  #Y_train[:,i,0] = np.zeros_like(y_train)
#  Y_train[:,i] = np.zeros_like(y_train)
#  #Y_test[:,i,0] = np.zeros_like(y_test)
#  Y_test[:,i] = np.zeros_like(y_test)
#  #Y_test_test[:,i,0] = np.zeros_like(y_test_test)
#  Y_test_test[:,i] = np.zeros_like(y_test_test)
##ratio = Y_train[:,:,1].sum()*1./(trY.shape[0]*6*6)*1.
#ratio = Y_train[:,:].sum()*1./(trY.shape[0]*6*6)*1.
#weights = np.array((ratio, 1-ratio))
#print('tr my mil ratio'+str(ratio))
# 将标签数据转换为one-hot编码格式
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_test_test = np_utils.to_categorical(y_test_test, nb_classes)
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'val samples')
print(X_test_test.shape[0], 'test samples')
# 模型构建
# 如果模型是 alexnet，则将单通道图像扩展为三通道（AlexNet 需要三通道输入
# 如果使用预训练模型，则加载预训练权重
model = Sequential()
if modelname == 'alexnet':
  X_train_extend = np.zeros((X_train.shape[0],3, 227, 227))
  for i in xrange(X_train.shape[0]):
    rex = np.resize(X_train[i,:,:,:], (227, 227))
    X_train_extend[i,0,:,:] = rex
    X_train_extend[i,1,:,:] = rex
    X_train_extend[i,2,:,:] = rex
  X_train = X_train_extend
  X_test_extend = np.zeros((X_test.shape[0], 3,227, 227))
  for i in xrange(X_test.shape[0]):
    rex = np.resize(X_test[i,:,:,:], (227, 227))
    X_test_extend[i,0,:,:] = rex
    X_test_extend[i,1,:,:] = rex
    X_test_extend[i,2,:,:] = rex
  X_test = X_test_extend
  X_test_test_extend = np.zeros((X_test_test.shape[0], 3, 227, 227))
  for i in xrange(X_test_test.shape[0]):
    rex = np.resize(X_test_test[i,:,:,:], (227,227))
    X_test_test_extend[i,0,:,:] = rex
    X_test_test_extend[i,1,:,:] = rex
    X_test_test_extend[i,2,:,:] = rex
  X_test_test = X_test_test_extend
  if pretrain:  # 227*227
    alexmodel = convnet('alexnet', weights_path='alexnet_weights.h5', heatmap=False, l1=l1factor, l2=l2factor)
    model = convnet('alexnet', outdim=2, l1=l1factor, l2=l2factor, sparsemil=sparsemil, sparsemill1=sparsemill1, sparsemill2=sparsemill2)
    for layer, mylayer in zip(alexmodel.layers, model.layers):
      print(layer.name)
      if mylayer.name == 'mil_1':
        break
      else:
        weightsval = layer.get_weights()
        print(len(weightsval))
        mylayer.set_weights(weightsval)
  else:
    model = convnet('alexnet', outdim=2, l1=l1factor,l2=l2factor, sparsemil=sparsemil, sparsemill1=sparsemill1, sparsemill2=sparsemill2)

# let's train the model using SGD + momentum (how original).
# 模型编译 使用Adam优化器编译模型，损失函数为分类交叉熵，评估指标为准确率
sgd = Adam(lr=lr) #SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.load_weights("alexnet_fd4_vf1_lr5e-05_l25e-06_l10_ep500_bs80_wFalse_drFalse50True_spTrue1e-050.0acc0.916666666667ep241.hdf5")
model.compile(loss='categorical_crossentropy', #categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])#, AUCEpoch,PrecisionEpoch,RecallEpoch,F1Epoch])
print(model.summary())
#filepath = savename+'-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5' #-{val_auc:.2f}-\
#{val_prec:.2f}-{val_reca:.2f}-{val_f1:.2f}.hdf5'
#checkpoint0 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
#checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# 回调函数 用于在训练过程中记录损失，准确率，AUC，精确率，召回率和F1分数
checkpoint0 = LossEpoch(savename, validation_data=(X_test, Y_test), interval=1)
checkpoint1 = ACCEpoch(savename, validation_data=(X_test, Y_test), interval=1)
checkpoint2 = AUCEpoch(savename, validation_data=(X_test, Y_test), interval=1)
checkpoint3 = PrecisionEpoch(savename, validation_data=(X_test, Y_test), interval=1)
checkpoint4 = RecallEpoch(savename, validation_data=(X_test, Y_test), interval=1)
checkpoint5 = F1Epoch(savename, validation_data=(X_test, Y_test), interval=1)
#checkpoint2 = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
#checkpoint3 = ModelCheckpoint(filepath, monitor='val_prec', verbose=1, save_best_only=True, mode='max')
#checkpoint4 = ModelCheckpoint(filepath, monitor='val_reca', verbose=1, save_best_only=True, mode='max')
#checkpoint5 = ModelCheckpoint(filepath, monitor='val_f1', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint0, checkpoint1, checkpoint2, checkpoint3, checkpoint4, checkpoint5]
#callbacks_list = [AUCEpoch, PrecisionEpoch, RecallEpoch, F1Epoch, checkpoint0, checkpoint1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
# 数据增强
# 如果启用数据增强，则使用ImageDataGenerator对训练数据进行实时增强
# 如果启用类别权重，则在训练时使用类别权重
if not data_augmentation:
  print('Not using data augmentation.')
  model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
  print('Using real-time data augmentation.')
  # this will do preprocessing and realtime data augmentation
  datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=45.0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,
        zerosquare=True,
        zerosquareh=noises,
        zerosquarew=noises,
        zerosquareintern=0.0)  # randomly flip images
  # compute quantities required for featurewise normalization
  # (std, mean, and principal components if ZCA whitening is applied)
  datagen.fit(X_train)
  # fit the model on the batches generated by datagen.flow()
  if weighted:
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test),
                        callbacks=callbacks_list,
                        class_weight=[weights[0], weights[1]])
  else:
    #print(Y_train.shape)
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test),
                        callbacks=callbacks_list)
