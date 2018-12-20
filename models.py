from __future__ import division
import os, argparse
import _pickle as cpickle
import tensorflow as tf
import numpy as np

tfe=tf.contrib.eager
tf.enable_eager_execution()

class DeterministicNetwork(tf.keras.Model):
    def __init__(self,opt,parser):
        super(DeterministicNetwork,self).__init__()
        self.opt = opt
        self.parser=parser
        self.ncond = parser.ncond
        self.npred = parser.npred
        self.nInHidden= parser.nInHidden
        self.nState = parser.nState
        self.nAction = parser.nAction
        self.batch_size = parser.batch_size
        self.nfeature = parser.nfeature
        self.LossType = parser.loss
        print("==========================================")
        print("Using MLP EEN DETERMINISTIC")
        print("==========================================")

        self.DEncoder = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nfeature),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU()
                    ]
                )
        self.DDecoder = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nState*self.npred),
                        tf.keras.layers.Reshape((self.npred,self.nState,1))
                    ]
                )

    def predict(self,Input):
        tmp = self.DEncoder(Input)
        return self.DDecoder(tmp)

    def compute_loss(self,Input,Target):
        Predict = self.predict(Input)
        if self.LossType == "l2":
            los = tf.keras.losses.mean_squared_error(Target,Predict)
        elif self.LossType=="l1":
            los = tf.keras.losses.mean_absolute_error(Target,Predict)
        elif self.LossType=="msle":
            los = tf.keras.losses.msle(Target,Predict)
        elif self.LossType=="mape":
            los = tf.keras.losses.mape(Target,Predict)
        return los

    def compute_gradients(self,Input,Target):
        with tf.GradientTape() as tape:
            L = self.compute_loss(Input,Target)
        return tape.gradient(L,self.trainable_variables),L

    def apply_gradients(self,grad,global_step=None):
        self.opt.apply_gradients(zip(grad,self.trainable_variables),global_step=global_step)


class LatentNetwork(tf.keras.Model):
    def __init__(self,opt,parser):
        super(LatentNetwork,self).__init__()
        self.opt = opt
        self.parser=parser
        self.ncond = parser.ncond
        self.npred = parser.npred
        self.nInHidden= parser.nInHidden
        self.nState = parser.nState
        self.nAction = parser.nAction
        self.batch_size = parser.batch_size
        self.nfeature = parser.nfeature
        self.LossType = parser.loss

        self.nLatent = parser.n_latent
        print("==========================================")
        print("Using MLP EEN DETERMINISTIC")
        print("==========================================")

        self.DEncoder = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nfeature),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU()
                    ]
                )
        self.DDecoder = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nState*self.npred),
                        tf.keras.layers.Reshape((self.npred,self.nState,1))
                    ]
                )

        self.phi = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(),
                    tf.keras.layers.Dense(self.nLatent,activation="tanh")
                ],name="phi"

           )
        self.LEncoder = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nfeature),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU()
                    ]
                )
        self.wz = tf.keras.layers.Dense(self.nfeature)
        self.LDecoder = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nInHidden),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.LeakyReLU(),
                        tf.keras.layers.Dense(self.nState*self.npred),
                        tf.keras.layers.Reshape((self.npred,self.nState,1))
                    ]
                )
        self._freeze_weight(self.DDecoder)
        self._freeze_weight(self.DEncoder)

    def _freeze_weight(self,Model):
        for layer in Model.layers:
            layer.trainable = False


    def decode(self,Input,z):
        f1 = self.LEncoder(Input)
        Wz = self.wz(z)
        return self.LDecoder(f1+Wz)

    def Dpredict(self,Input):
        tmp =self.DEncoder(Input)
        return self.DDecoder(tmp)

    def loss(self,Predict,Target):
        if self.LossType == "l2":
            los = tf.keras.losses.mean_squared_error(Target,Predict)
        elif self.LossType=="l1":
            los = tf.keras.losses.mean_absolute_error(Target,Predict)
        elif self.LossType=="msle":
            los = tf.keras.losses.msle(Target,Predict)
        elif self.LossType=="mape":
            los = tf.keras.losses.mape(Target,Predict)
        return los

    def compute_loss(self,Input,Target):
        Dpred = self.Dpredict(Input)
        rError = Dpred - Target
        z= self.phi(rError)
        f1 = self.LEncoder(Input)
        Wz = self.wz(z)

        Predict = self.LDecoder(f1+Wz)
        if self.LossType == "l2":
            los = tf.keras.losses.mean_squared_error(Target,Predict)
        elif self.LossType=="l1":
            los = tf.keras.losses.mean_absolute_error(Target,Predict)
        elif self.LossType=="msle":
            los = tf.keras.losses.msle(Target,Predict)
        elif self.LossType=="mape":
            los = tf.keras.losses.mape(Target,Predict)

        return los,Predict,Dpred,z

    def compute_gradients(self,Input,Target):
        with tf.GradientTape() as tape:
            L ,Predict,Dpredict,z= self.compute_loss(Input,Target)
        return tape.gradient(L,self.trainable_variables),L

    def apply_gradients(self,grad,global_step=None):
        self.opt.apply_gradients(zip(grad,self.trainable_variables),global_step=global_step)
