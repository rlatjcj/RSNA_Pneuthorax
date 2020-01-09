import keras
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Concatenate
from keras.layers import UpSampling2D
from keras.layers import Add
from keras.layers import Lambda
from keras.layers import Multiply
from keras.layers import Input
from keras.layers import Activation

from keras.models import Model

from .layers import _normalization, _activation, _downsizing, _se_block
from losses import *
from metrics import *

class MyModel:
    def __init__(self, args=None, axis=-1, noise=0.1, **kwargs):
        if args:
            self.fpn = None
            for k, v in vars(args).items():
                setattr(self, k, v)

            self.is_seblock = True if 'se' in self.bottom_up else False
            self.base_filter = int(64/self.divide)
            self.axis = axis
            self.noise = noise
            self.se_ratio = 16
        else:
            raise ValueError()

        self.mymodel = self.Unet()

    def _set_loss(self):
        if self.lossfn == 'dice':
            self.loss = dice_loss
        elif self.lossfn == 'dicewo':
            self.loss = dice_loss_wo
        elif self.lossfn == 'crossentropy':
            self.loss = crossentropy
        elif self.lossfn == 'focal':
            self.loss = focal()
        elif self.lossfn == 'cedice':
            self.loss = ce_dice_loss
        elif self.lossfn == 'focaldice':
            self.loss = focal_dice_loss
        elif self.lossfn == 'focaldicewo':
            self.loss = focal_dicewo_loss
        elif self.lossfn == 'celogdice':
            self.loss = ce_logdice_loss
        elif self.lossfn == 'focallogdice':
            self.loss = focal_logdice_loss

    def _set_optimizer(self):
        if self.optimizer == "adam":
            from keras.optimizers import Adam
            self.optim = Adam(lr=self.lr, clipnorm=.001)
        elif self.optimizer == "sgd":
            from keras.optimizers import SGD
            self.optim = SGD(lr=self.lr)

    def _set_metrics(self):
        self.metrics = [dice]
        if self.classes > 1:
            self.metrics.append(dice_pneu)

    def compile(self):
        self._set_loss()
        self._set_optimizer()
        self._set_metrics()

        self.mymodel.compile(
            optimizer=self.optim,
            loss=self.loss,
            metrics=self.metrics)

    def _set_backbone(self):
        if 'unet' in self.bottom_up:
            from keras.layers import Input
            img_input = Input(shape=(self.img_size, self.img_size, 1))
            if K.learning_phase() == 1:
                img_input = GaussianNoise(0.1)(img_input)

            c1 = self._basic_block(img_input, self.base_filter)
            c1 = self._basic_block(c1, self.base_filter)

            c2 = _downsizing(c1, self.base_filter, downsizing=self.downsizing, norm=self.norm, activation=self.activation)
            c2 = self._basic_block(c2, self.base_filter*2)
            c2 = self._basic_block(c2, self.base_filter*2)

            c3 = _downsizing(c2, self.base_filter*2, downsizing=self.downsizing, norm=self.norm, activation=self.activation)
            c3 = self._basic_block(c3, self.base_filter*4)
            c3 = self._basic_block(c3, self.base_filter*4)

            c4 = _downsizing(c3, self.base_filter*4, downsizing=self.downsizing, norm=self.norm, activation=self.activation)
            c4 = self._basic_block(c4, self.base_filter*8)
            c4 = self._basic_block(c4, self.base_filter*8)

            c5 = _downsizing(c4, self.base_filter*8, downsizing=self.downsizing, norm=self.norm, activation=self.activation)
            c5 = self._basic_block(c5, self.base_filter*16)
            c5 = self._basic_block(c5, self.base_filter*16)

            model = Model([img_input], [c1, c2, c3, c4, c5])
            return model

        elif 'res50' in self.bottom_up:
            from .resnet50 import ResNet50
            return ResNet50(
                input_shape=(self.img_size, self.img_size, self.channel),
                norm=self.norm,
                activation=self.activation,
                downsizing=self.downsizing,
                is_seblock=self.is_seblock,
                classes=self.classes
            )

        elif 'modixception' in self.bottom_up:
            from .modi_xception import Xception_modified
            return Xception_modified(
                input_shape=(self.img_size, self.img_size, self.channel),
                norm=self.norm,
                activation=self.activation,
                downsizing=self.downsizing,
                divide=self.divide,
                is_seblock=self.is_seblock
            )

        elif 'xception' in self.bottom_up:
            from .xception import Xception
            return Xception(
                input_shape=(self.img_size, self.img_size, self.channel),
                norm=self.norm,
                activation=self.activation,
                downsizing=self.downsizing,
                divide=self.divide,
                is_seblock=self.is_seblock
            )

        elif 'efficientb4' in self.bottom_up:
            from .efficientnet import EfficientNetB4
            return EfficientNetB4(
                input_shape=(self.img_size, self.img_size, self.channel))

        elif 'efficientb5' in self.bottom_up:
            from .efficientnet import EfficientNetB5
            return EfficientNetB5(
                input_shape=(self.img_size, self.img_size, self.channel))

        elif 'efficientb6' in self.bottom_up:
            from .efficientnet import EfficientNetB6
            return EfficientNetB6(
                input_shape=(self.img_size, self.img_size, self.channel))

        elif 'efficientb7' in self.bottom_up:
            from .efficientnet import EfficientNetB7
            return EfficientNetB7(
                input_shape=(self.img_size, self.img_size, self.channel))

    def Transfer(self):
        u1 = Conv2D(self.classes, (1, 1), strides=(1, 1), padding='same')(self.mymodel.get_layer('activation_57').output)
        u1_output = _activation(u1, activation='softmax', name='result')
        new_output = [u1_output]
        
        new_fpn_output = self.mymodel.get_layer('fpn').layers[12].output
        new_fpn_output = Conv2D(self.classes, (1, 1), strides=(1, 1), padding='same')(new_fpn_output)
        new_fpn_output = _activation(new_fpn_output, activation='softmax')
        new_fpn = Model(self.mymodel.get_layer('fpn').get_input_at(0), new_fpn_output, name='fpn')

        
        if '3' in self.fpn:
            uplist = [self.mymodel.get_layer('activation_65').output]
        else:
            # if '2' in self.fpn:
            #     uplist.append(self.mymodel.get_layer('activation_65').output)
            # if '3' in self.fpn:
            #     uplist.append(self.mymodel.get_layer('activation_66').output)
            # if '4' in self.fpn:
            #     uplist.append(self.mymodel.get_layer('activation_67').output)
            # if '5' in self.fpn:
            #     uplist.append(self.mymodel.get_layer('activation_68').output)
            # if len(uplist) < 1:
            uplist = [self.mymodel.get_layer('activation_65').output,
                      self.mymodel.get_layer('activation_66').output,
                      self.mymodel.get_layer('activation_67').output,
                      self.mymodel.get_layer('activation_68').output]

        for up in uplist:
            fpn_output = new_fpn(up)
            new_output.append(fpn_output)

        self.mymodel = Model([self.mymodel.input], new_output, name='{}-{}-{}'.format(self.bottom_up, self.top_down, self.skip))

    def Unet(self):
        backbone = self._set_backbone()

        img_input = backbone.input

        if 'modixception' in self.bottom_up or 'efficient' in self.bottom_up:
            if 'modixception' in self.bottom_up:
                d1, d2, d3, d4, d5 = backbone.output
            elif 'efficientb4' in self.bottom_up:
                d1, d2, d3, d4, d5 = [backbone.layers[l].output for l in [30, 92, 154, 342, 498]]
            elif 'efficientb5' in self.bottom_up:
                d1, d2, d3, d4, d5 = [backbone.layers[l].output for l in [43, 121, 199, 419, 607]]
            elif 'efficientb6' in self.bottom_up:
                d1, d2, d3, d4, d5 = [backbone.layers[l].output for l in [43, 137, 231, 483, 703]]
            elif 'efficientb7' in self.bottom_up:
                d1, d2, d3, d4, d5 = [backbone.layers[l].output for l in [56, 166, 276, 592, 860]]

            u5 = self._upconv_block(inputs=d5, 
                                    skip=[d4, d3, d2, d1] if 'dense' in self.skip else d4, 
                                    filters=self.base_filter*16)
            u4 = self._upconv_block(inputs=u5, 
                                    skip=[d5, d4, d3, d2, d1] if 'dense' in self.skip else d3, 
                                    filters=self.base_filter*8)
            u3 = self._upconv_block(inputs=u4, 
                                    skip=[u5, d5, d4, d3, d2, d1] if 'dense' in self.skip else d2, 
                                    filters=self.base_filter*4)
            u2 = self._upconv_block(inputs=u3, 
                                    skip=[u4, u5, d5, d4, d3, d2, d1] if 'dense' in self.skip else d1, 
                                    filters=self.base_filter*2)

            u1 = Conv2DTranspose(self.base_filter, (3, 3), strides=(2, 2), use_bias=False, padding="same")(u2)
            u1 = _normalization(u1, norm=self.norm)
            u1 = _activation(u1, activation=self.activation)

        elif 'unet' in self.bottom_up:
            d1, d2, d3, d4, d5 = backbone.output

            u4 = self._upconv_block(inputs=d5, 
                                    skip=[d4, d3, d2, d1] if 'dense' in self.skip else d4, 
                                    filters=self.base_filter*8)
            u3 = self._upconv_block(inputs=u4, 
                                    skip=[d5, d4, d3, d2, d1] if 'dense' in self.skip else d3, 
                                    filters=self.base_filter*4)
            u2 = self._upconv_block(inputs=u3, 
                                    skip=[u4, d5, d4, d3, d2, d1] if 'dense' in self.skip else d2, 
                                    filters=self.base_filter*2)
            u1 = self._upconv_block(inputs=u2, 
                                    skip=[u3, u4, d5, d4, d3, d2, d1] if 'dense' in self.skip else d1, 
                                    filters=self.base_filter)

        if self.fpn:
            fpn_model = self.FPN()
            shape_input = K.int_shape(img_input)

            u1 = Conv2D(self.classes, (1, 1), strides=(1, 1), padding='same')(u1)
            # u1 = _normalization(u1, norm=self.norm)
            u1_output = _activation(u1, activation='softmax', name='result')
            img_output = [u1_output]

            uplist = []
            if '2' in self.fpn:
                uplist.append(u2)
            if '3' in self.fpn:
                uplist.append(u3)
            if '4' in self.fpn:
                uplist.append(u4)
            if '5' in self.fpn:
                uplist.append(u5)
            if len(uplist) < 1:
                uplist = [u2, u3, u4, u5]

            for up in uplist:
                shape_up = K.int_shape(up)
                up = UpSampling2D(size=(shape_input[1]//shape_up[1], shape_input[2]//shape_up[2]))(up)
                up = Conv2D(int(256//self.divide), (1, 1), strides=(1, 1), padding='same')(up)
                up = _normalization(up, norm=self.norm)
                up = _activation(up, activation=self.activation)
                up_output = fpn_model(up)
                img_output.append(up_output)

            if self.last_relu == True:
                img_output = [ThresholdedReLU(theta=0.5)(io) for io in img_output.items()]

        else:
            img_output = Conv2D(self.classes, (1, 1), strides=(1, 1), padding='same')(u1)
            if self.checkpoint and 'Wed_Aug_21' not in self.checkpoint:
                # 수요일까지의 모델에는 없음
                img_output = _normalization(img_output, norm=self.norm)
            img_output = _activation(img_output, activation='softmax', name='result')

            if self.last_relu == True:
                img_output = ThresholdedReLU(theta=0.5)(img_output)

        model = Model(img_input, img_output, name='{}-{}-{}'.format(self.bottom_up, self.top_down, self.skip))
        
        return model

    def FPN(self):
        img_input = Input(shape=(None, None, int(256//self.divide)))

        x = img_input
        if 'direct' in self.fpn:
            pass
        elif 'unet' in self.fpn:
            for i in range(2):
                x = self._basic_block(x, int(256//self.divide))
                x = self._basic_block(x, int(256//self.divide))

        elif 'res' in self.fpn:
            for i in range(2):
                x = self._residual_block(x, int(256//self.divide))
        
        img_output = Conv2D(self.classes, (1, 1), strides=(1, 1), padding='same')(x)
        # img_output = _normalization(img_output, norm=self.norm)
        img_output = _activation(img_output, activation='softmax')

        model = Model(img_input, img_output, name='fpn')

        return model


    def _FDN(self, inputs, skip, filters):
        result = []
        shape_input = K.int_shape(inputs)
        for s in skip:
            s = Conv2D(filters//len(skip), (1, 1), strides=(1, 1), use_bias=False, padding='same')(s)
            s = _normalization(s, norm=self.norm)
            s = _activation(s, activation=self.activation)
            shape_s = K.int_shape(s)
            if shape_s[1] > shape_input[1]:
                s = Conv2D(filters//len(skip), (3, 3), strides=(shape_s[1]//(shape_input[1]*2), shape_s[2]//(shape_input[2]*2)), use_bias=False, padding='same')(s)
            else:
                s = Conv2DTranspose(filters//len(skip), (3, 3), strides=((shape_input[1]*2)//shape_s[1], (shape_input[2]*2)//shape_s[2]), use_bias=False, padding="same")(s)
            s = _normalization(s, norm=self.norm)
            s = _activation(s, activation=self.activation)
            result.append(s)
        
        return result

    def _upconv_block(self, inputs, skip, filters):
        if 'attention' in self.skip:
            skip = [self._atten_gate(inputs, skip, filters)]
        elif 'dense' in self.skip:
            skip = self._FDN(inputs, skip, filters)
        else:
            skip = [skip]

        x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), use_bias=False, padding="same")(inputs)
        x = _normalization(x, norm=self.norm)
        x = _activation(x, activation=self.activation)
        
        x = Concatenate()([x]+skip)

        if self.top_down == 'unet':
            x = self._basic_block(x, filters)
            x = self._basic_block(x, filters)

        elif self.top_down == 'res':
            x = self._residual_block(x, filters)

        return x

    def _atten_gate(self, inputs, skip, filters):
        def __expend_as(tensor, rep):
            my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
            return my_repeat

        gating = Conv2D(K.int_shape(inputs)[-1], (1, 1), use_bias=False, padding='same')(inputs)
        gating = _normalization(gating, norm=self.norm)
        shape_skip = K.int_shape(skip)
        shape_gating = K.int_shape(gating)

        #
        theta = Conv2D(filters, (2, 2), strides=(2, 2), use_bias=False, padding='same')(skip)
        shape_theta = K.int_shape(theta)

        phi = Conv2D(filters, (1, 1), use_bias=False, padding='same')(gating)
        phi = Conv2DTranspose(filters, (3, 3), 
                              strides=(shape_theta[1]//shape_gating[1], shape_theta[2]//shape_gating[2]), 
                              padding='same')(phi)

        add_xg = Add()([phi, theta])
        act_xg = _activation(add_xg, activation='relu')
        psi = Conv2D(1, (1, 1), use_bias=False, padding='same')(act_xg)
        sigmoid_xg = _activation(psi, activation='sigmoid')
        shape_sigmoid = K.int_shape(sigmoid_xg)

        upsample_psi = UpSampling2D(size=(shape_skip[1]//shape_sigmoid[1], shape_skip[2]//shape_sigmoid[2]))(sigmoid_xg)
        upsample_psi = __expend_as(upsample_psi, shape_skip[3])
        result = Multiply()([upsample_psi, skip])
        result = Conv2D(shape_skip[3], (1, 1), padding='same')(result)
        result = _normalization(result, norm=self.norm)
        return result

    def _basic_block(self, inputs, filters):
        x = Conv2D(filters, (3, 3), strides=(1, 1), use_bias=False, padding='same')(inputs)
        x = _normalization(x, norm=self.norm)
        x = _activation(x, activation=self.activation)
        return x

    def _residual_block(self, inputs, filters):
        residual = inputs

        x = Conv2D(filters//4, (1, 1), strides=(1, 1), use_bias=False, padding='same')(inputs)
        x = _normalization(x, norm=self.norm)
        x = _activation(x, activation=self.activation)

        x = Conv2D(filters//4, (3, 3), strides=(1, 1), use_bias=False, padding='same')(x)
        x = _normalization(x, norm=self.norm)
        x = _activation(x, activation=self.activation)

        x = Conv2D(filters, (1, 1), strides=(1, 1), use_bias=False, padding='same')(inputs)
        x = _normalization(x, norm=self.norm)

        if 'se' in self.top_down:
            x = _se_block(x, filters)

        residual = Conv2D(filters, (3, 3), strides=(1, 1), use_bias=False, padding='same')(residual)
        residual = _normalization(residual, norm=self.norm)

        x = Add()([x, residual])
        x = _activation(x, activation=self.activation)
        
        return x