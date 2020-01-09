import os
import sys
import json
import time
import argparse

import keras
import tensorflow as tf

from load_data import *
from models import *
from callbacks import *

sys.path.append(".")

def set_cbdir(args, stamp):
    if not os.path.isdir('./result/{}'.format(args.mode)):
        os.mkdir('./result/{}'.format(args.mode))

    if not os.path.isdir('./result/{}/{}'.format(args.mode, stamp)):
        os.mkdir('./result/{}/{}'.format(args.mode, stamp))
    elif not args.checkpoint:
        time.sleep(5)
        stamp = time.strftime("%c").replace(":", "_").replace(" ", "_")
        os.mkdir('./result/{}/{}'.format(args.mode, stamp))
    
    for i in ['checkpoint', 'history', 'logs']:
        if not os.path.isdir('./result/{}/{}/{}'.format(args.mode, stamp, i)):
            os.mkdir('./result/{}/{}/{}'.format(args.mode, stamp, i))
    
    with open('./result/{}/{}/model_desc.json'.format(args.mode, stamp), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))
    
    return stamp

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def check_args(parsed_args):
    return parsed_args

def get_arguments(args):
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='mode')
    subparser.required = True

    # segmentation
    segmentation_parser = subparser.add_parser('segmentation')
    segmentation_parser.add_argument("--bottom-up", type=str, default="unet", metavar="unet / xcep + se")
    segmentation_parser.add_argument("--top-down", type=str, default="unet", metavar="unet / res + se")
    segmentation_parser.add_argument("--skip", type=str, default="unet", metavar="unet / attention / dense")
    segmentation_parser.add_argument("--fpn", type=str, default=None, metavar="direct / basic / res")
    segmentation_parser.add_argument("--downsizing", type=str, default="pooling", metavar="pooling / conv")
    segmentation_parser.add_argument("--activation", type=str, default="relu", metavar="relu / leakyrelu")
    segmentation_parser.add_argument("--norm", type=str, default="bn", metavar="bn / in")
    segmentation_parser.add_argument("--divide", type=float, default=1.)
    segmentation_parser.add_argument("--last-relu", type=int, default=0)

    classification_parser = subparser.add_parser('classification')
    
    # hyper-parameter
    parser.add_argument("--lossfn", type=str, default="dice", metavar="dice / dicewo / crossentropy / focal / cedice / focaldice")
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-mode", type=str, default='exponential')
    parser.add_argument("--lr-value", type=float, default=.9)
    parser.add_argument("--lr-duration", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--standard", type=str, default="minmax", metavar="minmax / norm / eachnorm")
    parser.add_argument("--img-size", type=int, default=256)
    
    parser.add_argument("--summary", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--callback", type=int, default=1)
    parser.add_argument("--fp-path", type=str, default='Sat_Aug_24_05_33_27_2019')
    parser.add_argument("--whole", type=int, default=0)
    parser.add_argument("--transfer", type=int, default=0)

    return check_args(parser.parse_args(args))

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = get_arguments(args)

    keras.backend.tensorflow_backend.set_session(get_session())

    for k, v in vars(args).items():
        print('{} : {}'.format(k, v))

    ##############################################
    # Set Model
    ##############################################
    model = MyModel(args)
    if args.checkpoint:
        model.mymodel.load_weights(args.checkpoint)
        print("Load weights successfully at {}".format(args.checkpoint))
        if args.transfer:
            model.classes = 3
            args.classes = 3
            model.Transfer()
            args.initial_epoch = 0
            args.stamp = time.strftime("%c").replace(":", "_").replace(" ", "_")
        else:
            args.initial_epoch = int(args.checkpoint.split('/')[-1].split('_')[-2])
            args.stamp = args.checkpoint.split('/')[-3]
    else:
        args.initial_epoch = 0
        args.stamp = time.strftime("%c").replace(":", "_").replace(" ", "_")
        
    print("Initial epoch :", args.initial_epoch)
    print("Stamp :", args.stamp)

    if args.summary:
        from keras.utils import plot_model
        plot_model(model.mymodel, to_file='./model.png', show_shapes=True)
        model.mymodel.summary()
        return

    model.compile()

    ##############################################
    # Set Dataset & Generator
    ##############################################
    trainset, valset = load(args)
    train_generator = Generator(trainset, args, 'train')
    val_generator = Generator(valset, args, 'validation', rotation_range=0., shuffle=False)

    steps_per_epoch = 2142 // args.batch_size
    validation_steps = len(valset) // args.batch_size

    ##############################################
    # Set Callbacks
    ##############################################
    callbacks = []
    if args.callback:
        args.stamp = set_cbdir(args, args.stamp)
        cp = callback_checkpoint(filepath=os.path.join('./result/{}/{}/checkpoint'.format(args.mode, args.stamp), '{epoch:04d}_{val_result_dice:.4f}.h5'),
                                monitor='val_result_dice',
                                verbose=1,
                                mode='max',
                                save_best_only=False,
                                save_weights_only=False)

        el = callback_epochlogger(filename=os.path.join('./result/{}/{}/history'.format(args.mode, args.stamp), 'epoch.csv'),
                                separator=',', append=True)

        bl = callback_batchlogger(filename=os.path.join('./result/{}/{}/history'.format(args.mode, args.stamp), 'batch.csv'),
                                separator=',', append=True)

        tb = callback_tensorboard(log_dir=os.path.join('./result/{}/{}/logs'.format(args.mode, args.stamp)), batch_size=args.batch_size)
        
        ls = callback_learningrate(initlr=args.lr, 
                                   mode=args.lr_mode,
                                   value=args.lr_value, 
                                   duration=args.lr_duration, 
                                   total_epoch=args.epochs)

        callbacks += [cp, el, bl, tb, ls]

    model.mymodel.fit_generator(generator=train_generator,
                                steps_per_epoch=steps_per_epoch,
                                verbose=1,
                                epochs=args.epochs,
                                validation_data=val_generator,
                                validation_steps=validation_steps,
                                callbacks=callbacks,
                                shuffle=True,
                                initial_epoch=args.initial_epoch)

if __name__ == "__main__":
    main()