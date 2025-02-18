#import dicom # some machines not install pydicom
import scipy.misc
import numpy as np 
from sklearn.model_selection import StratifiedKFold
import cPickle
#import matplotlib
#import matplotlib.pyplot as plt 
from skimage.filters import threshold_otsu
import os
from os.path import join as join
import csv
import scipy.ndimage
import dicom
#import cv2
path = '../AllDICOMs/'
preprocesspath = '../preprocesspath/'
labelfile = './label.txt'

# 从一个标签文件（labelfile）中读取标签信息，并将其存储为一个字典（mydict，确保标签值为0或1
def readlabel():
  '''read the label as a dict from labelfile'''
  mydict = {} # 初始化一个空字典
  with open(labelfile, 'r') as f: # 打开标签文件
    flines = f.readlines() # 读取文件的所有行
    for line in flines: # 遍历每一行
      data = line.split() # 将行按空格分割为列表
      if int(data[1]) == 0: # 如果标签值为0
        mydict[data[0]] = int(data[1]) # 直接存储为0
      else: # 如果标签值为1或2
        assert(int(data[1])==2 or int(data[1])==1) # 确保标签值为1或2
        mydict[data[0]] = int(data[1])-1 # 将标签值转换为0或1
  return mydict # 返回字典


def readdicom(mydict):
  '''read the dicom image, rename it consistently with the name in labels, crop and resize, and save as pickle.
  mydict is the returned value of readlabel'''
  img_ext = '.dcm' # DICOM文件扩展名
  img_fnames = [x for x in os.listdir(path) if x.endswith(img_ext)] # 图像名称
  for f in img_fnames: # 遍历每个文件
    names = f.split('_') # 将文件名按 _ 分割，提取前缀（names[0]）
    if names[0] not in mydict:
      print(names[0]+'occur error')
    dicom_content = dicom.read_file(join(path,f)) # 读取文件
    img = dicom_content.pixel_array # 获取图像像素数据
    '''fig = plt.figure()
    ax1 = plt.subplot(3,3,1)
    ax2 = plt.subplot(3,3,2)
    ax3 = plt.subplot(3,3,3)
    ax4 = plt.subplot(3,3,4)
    ax5 = plt.subplot(3,3,5)
    ax6 = plt.subplot(3,3,6)
    ax7 = plt.subplot(3,3,7)
    ax8 = plt.subplot(3,3,8)
    ax9 = plt.subplot(3,3,9)
    ax1.imshow(img, cmap='Greys_r')
    ax1.set_title('Original')
    ax1.axis('off')'''
    
    thresh = threshold_otsu(img) # 使用 Otsu 阈值法对图像进行二值化，生成二值图像 binary
    binary = img > thresh
    #ax2.imshow(binary, cmap='Greys_r')
    #ax2.set_title('mask')
    #ax2.axis('off')

    # 裁剪图像
    # 初始化裁剪区域的边界（minx，miny，maxx，maxy）
    minx, miny = 0, 0
    maxx, maxy = img.shape[0], img.shape[1]
    # 遍历图像的每一行，找到有效区域的上下边界minxx和maxx
    for xx in xrange(img.shape[1]):
      if sum(binary[xx, :]==0) < binary.shape[1]-60:
        minx = xx
        break
    for xx in xrange(img.shape[0]-1,0,-1):
      if sum(binary[xx, :]==0) < binary.shape[1]-60:
        maxx = xx
        break
    # 根据文件名中的信息（names[3]）判断图像是左侧还是右侧，找到有效区域的左右边界（miny 和 maxy）
    if names[3] == 'R':
      maxy = img.shape[1]
      for yy in xrange(int(img.shape[1]*3.0/4), -1, -1):
        if sum(binary[:,yy]==0) > binary.shape[0]-10: 
          miny = yy
          break
    else:
      miny = 0
      for yy in xrange(int(img.shape[1]/4.0), img.shape[1], 1):
        if sum(binary[:,yy]==0) > binary.shape[0]-10: 
          maxy = yy
          break
    print(minx, maxx, miny, maxy)
    #ax3.set_title('Foreground')
    #ax3.imshow(img[minx:maxx+1, miny:maxy+1], cmap='Greys_r')
    #ax3.axis('off')
    # 缩放图像
    img = img.astype(np.float32)
    # 将裁剪后的图像缩放到 227x227 和 299x299 大小
     # 使用 cPickle.dump 将缩放后的图像保存为 pickle 文件
    img1 = scipy.misc.imresize(img[minx:maxx+1, miny:maxy+1], (227, 227), interp='cubic')
    with open(join(preprocesspath, names[0])+'227.pickle', 'wb') as outfile:
      cPickle.dump(img1, outfile)
    img1 = scipy.misc.imresize(img[minx:maxx+1, miny:maxy+1], (299, 299), interp='cubic')
    with open(join(preprocesspath, names[0])+'299.pickle', 'wb') as outfile:
      cPickle.dump(img1, outfile) 
    '''ax4.set_title('Resize')
    ax4.imshow(img, cmap='Greys_r')
    ax4.axis('off')

    img = img.astype(np.float32)
    img -= np.mean(img)
    img /= np.std(img)
    ax5.set_title('Norm')
    ax5.imshow(img, cmap='Greys_r')
    ax5.axis('off')
    with open(join(preprocesspath, names[0])+'norm.pickle', 'wb') as outfile:
      cPickle.dump(img, outfile)
      #imgshape = img.shape
    
    img = np.fliplr(img)
    ax6.set_title('Flip')
    ax6.imshow(img, cmap='Greys_r')
    ax6.axis('off')
    
    num_rot = np.random.choice(4)               #rotate 90 randomly
    img = np.rot90(img, num_rot)
    ax7.set_title('Rotation')
    ax7.imshow(img, cmap='Greys_r')
    ax7.axis('off')
    fig.savefig(join(preprocesspath, names[0])+'.jpg')
    plt.close(fig)'''

# 实现分层K折交叉验证的数据划分
# 根据输入的 fold 和 totalfold 参数，将数据集划分为训练集和测试集
# 使用分层 K 折交叉验证，确保每个 fold 中各类别的比例与原始数据集一致
# 返回指定 fold 的训练集和测试集的索引
# 参数说明
# fold：当前需要返回的 fold 编号（从 0 到 totalfold-1）
# totalfold：交叉验证的总折数
# mydict：一个字典，键为样本标识（如文件名），值为标签。通常由 readlabel() 函数生成
def cvsplit(fold, totalfold, mydict):
  '''get the split of train and test
  fold is the returned fold th data, from 0 to totalfold-1
  total fold is for the cross validation
  mydict is the return dict from readlabel'''
  # 分层K折交叉验证
  # n_splits=totalfold：指定交叉验证的总折数
  # shuffle=False：默认不随机打乱数据（可以通过设置 shuffle=True 来打乱数据）
  skf = StratifiedKFold(n_splits=totalfold)  # default shuffle is false, okay!
  #readdicom(mydict)
  # 准备数据
  y = mydict.values() # 标签列表
  x = mydict.keys() # 样本标识列表
  # 划分数据
  count = 0
  for train, test in skf.split(x,y): # 遍历 StratifiedKFold 生成的划分结果
    print(len(train), len(test))
    if count == fold: 
      # 如果当前 fold 编号（count）等于目标 fold 编号（fold），则返回当前 fold 的训练集和测试集的索引
      #print test
      return train, test # 输出：训练集和测试集的索引列表
    # 否则继续遍历下一个fold
    count += 1

# 增强版的分层K折交叉验证，将数据集划分为训练集、验证集和测试集
# fold：当前需要返回的 fold 编号（从 0 到 totalfold-1）
# totalfold：交叉验证的总折数
# mydict：一个字典，键为样本标识（如文件名），值为标签。通常由 readlabel() 函数生成
# valfold：验证集的 fold 编号。如果未指定（-1），则默认为 (fold + 1) % totalfold
def cvsplitenhance(fold, totalfold, mydict, valfold=-1):
  '''get the split of train and test
  fold is the returned fold th data, from 0 to totalfold-1
  total fold is for the cross validation
  mydict is the return dict from readlabel
  sperate the data into train, validation, test'''
  skf = StratifiedKFold(n_splits=totalfold)  # default shuffle is false, okay!
  #readdicom(mydict)
  y = mydict.values()
  x = mydict.keys()
  count = 0
  # 验证集fold编号
  # 如果未指定验证集 fold 编号（valfold=-1），则默认为 (fold + 1) % totalfold
  if valfold == -1: 
    valfold = (fold+1) % totalfold
  print('valfold'+str(valfold)) # 打印验证集fold编号
  # 划分数据
  trainls, valls, testls = [], [], []
  for train, test in skf.split(x,y):
    print(len(train), len(test))
    if count == fold: # 如果当前 fold 编号（count）等于目标 fold 编号（fold
      #print test[:] 
      testls = test[:] # 当前fold作为测试集
    elif count == valfold: # 当前 fold 编号等于验证集 fold 编号（valfold）
      valls = test[:] # 指定fold作为验证集
    else: # 其他fold的数据合并为训练集
      for i in test:
        trainls.append(i) # 其他fold作为训练集
    count += 1
  return trainls, valls, testls # 返回索引列表

def loadim(fname, preprocesspath=preprocesspath):
  ''' from preprocess path load fname
  fname file name in preprocesspath
  aug is true, we augment im fliplr, rot 4'''
  ims = []
  with open(join(preprocesspath, fname), 'rb') as inputfile:
    im = cPickle.load(inputfile)
    #up_bound = np.random.choice(174)                          #zero out square
    #right_bound = np.random.choice(174)
    img = im
    #img[up_bound:(up_bound+50), right_bound:(right_bound+50)] = 0.0
    ims.append(img)
    inputfile.close()
  return ims

def loaddata(fold, totalfold, usedream=True, aug=True):
  '''get the fold th train and  test data from inbreast
  fold is the returned fold th data, from 0 to totalfold-1
  total fold is for the cross validation'''
  mydict = readlabel()
  mydictkey = mydict.keys()
  mydictvalue = mydict.values()
  trainindex, testindex = cvsplit(fold, totalfold, mydict)
  if aug == True:
    traindata, trainlabel = np.zeros((6*len(trainindex),227,227)), np.zeros((6*len(trainindex),))
  else:
    traindata, trainlabel = np.zeros((len(trainindex),227,227)), np.zeros((len(trainindex),))
  testdata, testlabel =  np.zeros((len(testindex),227,227)), np.zeros((len(testindex),))
  traincount = 0
  for i in xrange(len(trainindex)):
    ims = loadim(mydictkey[trainindex[i]]+'.pickle', aug=aug)
    for im in ims:
      traindata[traincount, :, :] = im
      trainlabel[traincount] = mydictvalue[trainindex[i]]
      traincount += 1
  assert(traincount==traindata.shape[0])
  testcount = 0
  for i in xrange(len(testindex)):
    ims = loadim(mydictkey[testindex[i]]+'.pickle', aug=aug)
    testdata[testcount,:,:] = ims[0]
    testlabel[testcount] = mydictvalue[testindex[i]]
    testcount += 1
  assert(testcount==testdata.shape[0])
  if usedream:
    outx, outy = extractdreamdata()
    traindata = np.concatenate((traindata,outx), axis=0)
    trainlabel = np.concatenate((trainlabel,outy), axis=0)
  return traindata, trainlabel, testdata, testlabel

def loaddataenhance(fold, totalfold, valfold=-1, valnum=60):
  '''get the fold th train and  test data from inbreast
  fold is the returned fold th data, from 0 to totalfold-1
  total fold is for the cross validation'''
  mydict = readlabel()
  mydictkey = mydict.keys()
  mydictvalue = mydict.values()
  trainindex, valindex, testindex = cvsplitenhance(fold, totalfold, mydict, valfold=valfold)
  traindata, trainlabel = np.zeros((len(trainindex),227,227)), np.zeros((len(trainindex),))
  valdata, vallabel =  np.zeros((len(valindex),227,227)), np.zeros((len(valindex),))
  testdata, testlabel =  np.zeros((len(testindex),227,227)), np.zeros((len(testindex),))
  traincount = 0
  for i in xrange(len(trainindex)):
    ims = loadim(mydictkey[trainindex[i]]+'227.pickle')
    for im in ims:
      traindata[traincount, :, :] = im
      trainlabel[traincount] = int(mydictvalue[trainindex[i]])
      traincount += 1
  assert(traincount==traindata.shape[0])
  valcount = 0
  for i in xrange(len(valindex)):
    ims = loadim(mydictkey[valindex[i]]+'227.pickle')
    valdata[valcount,:,:] = ims[0]
    vallabel[valcount] = int(mydictvalue[valindex[i]])
    valcount += 1
  assert(valcount==valdata.shape[0])
  testcount = 0
  for i in xrange(len(testindex)):
    #print mydictkey[testindex[i]]
    ims = loadim(mydictkey[testindex[i]]+'227.pickle')
    testdata[testcount,:,:] = ims[0]
    testlabel[testcount] = int(mydictvalue[testindex[i]])
    testcount += 1
  assert(testcount==testdata.shape[0])
  #print(valdata.shape)
  randindex = np.random.permutation(valdata.shape[0])
  valdata = valdata[randindex,:,:]
  vallabel = vallabel[randindex]
  #print(valdata.shape)
  traindata = np.concatenate((traindata, valdata[valnum:,:,:]), axis=0)
  trainlabel = np.concatenate((trainlabel, vallabel[valnum:]), axis=0)
  valdata = valdata[:valnum,:,:]
  vallabel = vallabel[:valnum]
  maxvalue = (traindata.max()*1.0)
  print('inbreast max %f', maxvalue)
  traindata = traindata / maxvalue
  valdata = valdata / maxvalue
  testdata = testdata / maxvalue
  print('train data feature')
  #meanx = traindata.mean()
  #stdx = traindata.std()
  #traindata -= meanx
  #traindata /= stdx
  #valdata -= meanx
  #valdata /= stdx
  #testdata -= meanx
  #testdata /= stdx
  print(traindata.mean(), traindata.std(), traindata.max(), traindata.min())
  print('val data feature')
  print(valdata.mean(), valdata.std(), valdata.max(), valdata.min())
  print('test data feature')
  print(testdata.mean(), testdata.std(), testdata.max(), testdata.min())
  #meandata = traindata.mean()
  #stddata = traindata.std()
  #traindata = traindata - meandata
  #traindata = traindata / stddata
  #valdata = valdata - meandata
  #valdata = valdata / stddata
  #testdata = testdata - meandata
  #testdata = testdata / stddata
  return traindata, trainlabel, valdata, vallabel, testdata, testlabel

if __name__ == '__main__':
  traindata, trainlabel, testdata, testlabel = loaddata(0, 5)
  print(sum(trainlabel), sum(testlabel))

  traindata, trainlabel, valdata, vallabel, testdata, testlabel = loaddataenhance(0, 5)
  print(sum(trainlabel), sum(vallabel), sum(testlabel))
