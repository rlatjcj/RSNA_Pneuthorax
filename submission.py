import os
import sys
import json
import tqdm
import argparse
import pandas as pd

from load_data import *
from models import *

import keras
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1

    return " ".join(rle)

def get_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)

    return parser.parse_args(args)

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = get_arguments(args)

    keras.backend.tensorflow_backend.set_session(get_session())

    stamp = args.checkpoint.split('/')[-3]
    epoch = args.checkpoint.split('/')[-1].split('_')[0]
    with open('./result/segmentation/{}/model_desc.json'.format(stamp), 'r') as f:
        model_desc = json.load(f)
        for k, v in model_desc.items():
            if k == 'checkpoint':
                continue
            setattr(args, k, v)

    df = pd.read_csv('./stage_2_sample_submission.csv')
    testset = df['ImageId'].tolist()
    rles = []

    for k, v in vars(args).items():
        print('{} : {}'.format(k, v))

    model = MyModel(args)
    model.mymodel.load_weights(args.checkpoint)
    prep = Preprocessing(args, 0.)
    i = 0

    for t in tqdm.trange(len(testset)):
        data = testset[t]
        info = df[df['ImageId'] == data]
        imgo = pydicom.dcmread('/data/public/rw/kiminhwan/tdrw/data/test1/'+data+'.dcm').pixel_array.astype('float32')
        img = imgo.copy().astype('uint8')
        
        if args.img_size < 1024:
            img = prep._resize(img, args.img_size/1024)
        img = prep._standardize(img)
        img = prep._expand(img)
        # y_pred = np.argmax(np.squeeze(model.mymodel.predict_on_batch(img)), axis=-1)
        if args.fpn:
            if args.classes == 1:
                y_pred = np.squeeze(model.mymodel.predict_on_batch(img)[0][...,0])
            else:
                y_pred = np.squeeze(model.mymodel.predict_on_batch(img)[0][...,1])
        else:
            y_pred = np.squeeze(model.mymodel.predict_on_batch(img)[...,1])
        threshold = .7
        y_pred = (y_pred > threshold).astype('float32')
        if args.img_size < 1024:
            y_pred = prep._resize(y_pred, 1024/args.img_size)
            y_pred[y_pred > 0] = 1
        
        if args.classes == 1:
            y_pred = 1-y_pred

        rles.append(mask2rle(y_pred*255, 1024, 1024))
        if t < 5:
            import cv2
            cv2.imwrite('./result/{}.png'.format(t), imgo)
            cv2.imwrite('./result/{}_mask.png'.format(t), y_pred*255)
        
    submission = pd.DataFrame({'ImageId': testset, 'EncodedPIxels': rles})
    submission.loc[submission.EncodedPIxels=='', 'EncodedPIxels'] = '-1'
    submission.to_csv('./{}_{}.csv'.format(stamp, epoch), index=False)
    

if __name__ == "__main__":
    main()