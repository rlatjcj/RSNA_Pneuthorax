import os
import cv2
import tqdm
import random
import pydicom
import threading
import numpy as np
import pandas as pd

from scipy import ndimage

INPUT_FOLDER = '/data/public/rw/kiminhwan/tdrw/data/train/'

def load(args, seed=42, test_size=0.1):
    def count_annotation(datalist):
        cnt_anno = {}
        for tl in tqdm.trange(len(datalist)):
            info = df[df['ImageId'] == datalist[tl]]

            if len(info) > 1:
                if not len(info) in cnt_anno.keys():
                    cnt_anno[len(info)] = [1]
                    cnt_anno[len(info)].append(datalist[tl])
                else:
                    cnt_anno[len(info)][0] += 1
                    cnt_anno[len(info)].append(datalist[tl])
            else:
                if int(info[' EncodedPixels'].values[0].split()[0]) == -1:
                    if not 0 in cnt_anno.keys():
                        cnt_anno[0] = [1]
                        cnt_anno[0].append(datalist[tl])
                    else:
                        cnt_anno[0][0] += 1
                        cnt_anno[0].append(datalist[tl])
                else:
                    if not 1 in cnt_anno.keys():
                        cnt_anno[1] = [1]
                        cnt_anno[1].append(datalist[tl])
                    else:
                        cnt_anno[1][0] += 1
                        cnt_anno[1].append(datalist[tl])
        return cnt_anno

    if not os.path.isfile('./trainset_new.txt'):
        df = pd.read_csv('/data/public/rw/kiminhwan/tdrw/src/train-rle.csv')
        train_list = df['ImageId'].unique()
        whole_cnt = count_annotation(train_list)

        train_dict = {}
        val_dict = {}
        for k in sorted(whole_cnt.keys()):
            if int(whole_cnt[k][0]*test_size) == 0 and whole_cnt[k][0] > 1:
                print(k, whole_cnt[k][0], '-->', 1)
                val_dict[k] = whole_cnt[k][1:2]
                train_dict[k] = whole_cnt[k][2:]
            else:
                print(k, whole_cnt[k][0], '-->', int(whole_cnt[k][0]*test_size))
                val_dict[k] = whole_cnt[k][1:int(whole_cnt[k][0]*test_size)+1]
                train_dict[k] = whole_cnt[k][int(whole_cnt[k][0]*test_size)+1:]

        trainset = []
        valset = []
        for k, v in train_dict.items():
            if k == 0:
                continue
            for t in v:
                trainset.append(t)

        if args.mode == 'classification':
            for t in train_dict[0]:
                trainset.append(t)

        for k, v in val_dict.items():
            if k == 0:
                continue
            for t in v:
                valset.append(t)
        
        if args.mode == 'classification':
            for t in val_dict[0]:
                valset.append(t)

        with open('./trainset_{}.txt'.format(args.mode), 'w') as f:
            for t in trainset:
                f.write(t+'\n')
        with open('./valset_{}.txt'.format(args.mode), 'w') as f:
            for v in valset:
                f.write(v+'\n')

    else:
        # if args.whole:
        with open('./trainset_new.txt', 'r') as f:
            trainset = f.readlines()
            trainset = [t[:-1] for t in trainset]
        with open('./valset_classification.txt', 'r') as f:
            valset = f.readlines()
            valset = [v[:-1] for v in valset]
        # else:
        #     with open('./trainset_{}.txt'.format(args.mode), 'r') as f:
        #         trainset = f.readlines()
        #         trainset = [t[:-1] for t in trainset]
        #     with open('./valset_{}.txt'.format(args.mode), 'r') as f:
        #         valset = f.readlines()
        #         valset = [v[:-1] for v in valset]
            
    print('# of train data :', len(trainset), ', # of validation data :', len(valset))
    return trainset, valset


##############################################
# Generator for FINE
##############################################
class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def Generator(datalist,
              args,
              mode,
              rotation_range=10.,
              seed=42,
              shuffle=True,
              **kwargs):

    def _rle2mask(rle, width, height):
        mask= np.zeros(width* height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position+lengths[index]] = 255
            current_position += lengths[index]

        return mask.reshape(width, height)

    assert mode in ['train', 'validation', 'test']
    random.seed(seed)
    prep = Preprocessing(args, rotation_range=rotation_range)

    df = pd.read_csv('/data/public/rw/kiminhwan/tdrw/src/train-rle.csv')

    if mode in ['train', 'validation']:
        batch_size = args.batch_size
    else:
        batch_size = 1
    
    batch = 0
    X = np.empty((batch_size, args.img_size, args.img_size, args.channel))
    Y = np.empty((batch_size, args.img_size, args.img_size, args.classes))

    nomask = 0
    while True:
        if shuffle:
            random.shuffle(datalist)

        for data in datalist:
            if mode == 'test':
                print(data)

            info = df[df['ImageId'] == data]
            if len(info) == 0:
                continue

            mask = np.zeros((1024, 1024))
            if int(info[' EncodedPixels'].values[0].split()[0]) == -1:
                if args.whole:
                    if nomask >= batch_size//2 and mode == 'train':
                        continue
                    nomask += 1
                else:
                    pass
            else:
                for j in range(len(info)):
                    mask += _rle2mask(info.values[j,1], 1024, 1024).T

            img = pydicom.dcmread(INPUT_FOLDER+data+'.dcm').pixel_array.astype('float32')
            if args.img_size < 1024:
                img, mask = prep._resize(img, args.img_size/1024), prep._resize(mask, args.img_size/1024)
                mask[mask > 0.] = 1

            if mode == 'train':
                img, mask = prep._rotation([img, mask])
                img, mask = prep._horizontal_flip([img, mask])

            img = prep._standardize(img)
            mask = prep._merge_mask(mask)
            if args.classes == 1:
                mask = 1 - mask
                mask = mask[...,np.newaxis]
            else:
                mask = prep._onehot(mask)
                if args.classes == 3:
                    if os.path.isfile('/data/public/rw/kiminhwan/tdrw/data/png/train/{}/{}.png'.format(data, args.fp_path)):
                        fp = cv2.imread('/data/public/rw/kiminhwan/tdrw/data/png/train/{}/{}.png'.format(data, args.fp_path), cv2.IMREAD_GRAYSCALE)
                        fp = prep._resize(fp, args.img_size/1024)
                        mask[...,2] += fp
                        mask[...,0] -= fp
                
            img = prep._expand(img)
            X[batch] = img
            Y[batch] = mask
            batch += 1
            if batch >= batch_size:
                if args.fpn:
                    Y_output = [Y]
                    if '2' in args.fpn:
                        Y_output.append(Y)
                    if '3' in args.fpn:
                        Y_output.append(Y)
                    if '4' in args.fpn:
                        Y_output.append(Y)
                    if '5' in args.fpn:
                        Y_output.append(Y)
                    if len(Y_output) < 2:
                        Y_output = [Y, Y, Y, Y, Y]
                    
                    yield [X], Y_output
                else:
                    yield [X], [Y]
                
                if args.whole:
                    nomask = 0
                    
                batch = 0
                X = np.empty((batch_size, args.img_size, args.img_size, args.channel))
                Y = np.empty((batch_size, args.img_size, args.img_size, args.classes))


##############################################
# Preprocessing
##############################################
class Preprocessing:
    def __init__(self,
                 args,
                 rotation_range=10.):

        self.channel = args.channel
        self.standard = args.standard
        self.classes = args.classes
        self.rotation_range = rotation_range

    def _resize(self, x, ratio):
        x = ndimage.zoom(x, [ratio, ratio], order=1, mode='constant', cval=0.)
        return x

    def _expand(self, x):
        return x[np.newaxis,...] if self.channel > 1 else x[np.newaxis,...,np.newaxis]

    def _merge_mask(self, x):
        x[x != 0] = 1.
        return x

    def _onehot(self, x):
        result = np.zeros(x.shape+(self.classes,))
        for i in range(self.classes):
            result[...,i][np.where(x == i)] = 1.
        return result

    def _horizontal_flip(self, xx, axis=2):
        img, mask = xx
        if np.random.random() < 0.5:
            img = img[:,::-1]
            mask = mask[:,::-1]
        return img, mask

    def _standardize(self, x):
        if self.standard == "norm":
            return (x-109.91300761560832)/37.939738653752926
        elif self.standard == "eachnorm":
            return (x-x.mean())/x.std()
        elif self.standard == "minmax":
            return x/255
        else:
            return x

    def _rotation(self, xx, dep_index=0, row_index=1, col_index=2, fill_mode='nearest', cval=0.):
        img, mask = xx
        theta = np.random.uniform(-self.rotation_range, self.rotation_range)
        img = self.__apply_transform(img, theta=theta)
        mask = self.__apply_transform(mask, theta=theta)
        return img, mask

    def __transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix
        
    def __apply_transform(
        self, 
        x, 
        theta=0, 
        row_axis=0, 
        col_axis=1, 
        channel_axis=2, 
        fill_mode='nearest', 
        cval=0., 
        order=1):

        theta = np.deg2rad(theta)
        transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])

        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = self.__transform_matrix_offset_center(transform_matrix, h, w)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        x = ndimage.interpolation.affine_transform(
            x,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=cval)
        return x
        

    

if __name__ == "__main__":
    import sys
    from main import get_arguments
    args = get_arguments(sys.argv[1:])
    trainset, valset = load(args)

    gen = Generator(trainset, args, 'validation')
    for i in range(10):
        g = next(gen)
        print(i, g[0][0].shape, g[1][0].shape, g[0][0].min(), g[0][0].max(), np.unique(g[1][0]))
        if args.whole:
            print(g[1][0].sum(), g[1][0][0,...,0].sum(), g[1][0][0,...,1].sum(), g[1][0][1,...,0].sum(), g[1][0][1,...,1].sum())
        else:
            print(g[1][0].sum(), g[1][0][0,...,0].sum(), g[1][0][0,...,1].sum(), g[1][0][1,...,0].sum(), g[1][0][1,...,1].sum())