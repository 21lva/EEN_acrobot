import os,glob,argparse,csv
import numpy as np
import tensorflow as tf
import pandas as pd
from random import randrange as RI
import sklearn


class DataMaker():
    def __init__(self,parser):
        self.par = parser
        self.npred = parser.npred
        self.ncond = parser.ncond
        self.BatchSize = parser.batch_size
        self.DoValid = parser.valid
        self.n_trials = parser.n_trials
        self.env_id = parser.env_id
        self.test_size=parser.test_size
        self.train_size = parser.train_size
        self.TypeScaler = parser.ScalerType

        if self.TypeScaler =="StandardScaler":
            self.StateScaler = sklearn.preprocessing.StandardScaler()
            self.ActionScaler = sklearn.preprocessing.StandardScaler()

        else:
            self.StateScaler = sklearn.preprocessing.MinMaxScaler()
            self.ActionScaler = sklearn.preprocessing.MinMaxScaler()

        load_dir_path  = './expert_policies/saved_trajectories/trpo_' + self.env_id
        load_file_path = os.path.join(load_dir_path, self.env_id + "_" + str(self.n_trials) + "_trials.npz")
        print("=========================================")
        print("Loading observations in trajectories from the file: ", load_file_path)
        print("=========================================")
        self.ep= np.load(load_file_path)

        self.Box =[self.ep["obs"],self.ep["act"]]

        self.train_dataset=[self.ep["obs"][:self.train_size],self.ep["act"][:self.train_size]]
        if self.DoValid=="True":
            self.valid_dataset=[self.ep["obs"][self.train_size:],self.ep["act"][self.train_size:]]
        self.StateScaler.fit(np.vstack(self.ep["obs"]))
        self.ActionScaler.fit(np.vstack(self.ep["act"]))
        self.Box=self._transform()


    def _transform(self):
        res=[[],[]]
        for episode in self.Box[0]:
            res[0].append(self.StateScaler.transform(episode))
        for episode in self.Box[1]:
            res[1].append(self.ActionScaler.transform(episode))
        return res

    def get_one(self,Type,IsRandom=True):
        if Type=="train":
            Box=self.Box#self.train_dataset
            start = 0
            final = self.train_size
        elif Type=="valid":
            Box = self.Box#self.valid_dataset
            start =self.train_size
            final = self.n_trials-self.test_size
            IsRandom=False
        elif Type=="test":
            Box=self.Box
            start =self.n_trials-self.test_size
            final = self.n_trials
            IsRandom=False
        else:
            raise ValueError("    Only Three Types Availabe : train | valid | test")
        if Type=="train":
            C=[]
            B=[]
            D=[]
            #Box has shape (obs-act,episode ,state|action idx,state|action component)
            #NN  Input : (batch, ncond,state+action,1)
            #    return : (batch,npred,state,1)

            EpisodeIdx = RI(start,final)
            StartStateIdx = RI(0,Box[0][EpisodeIdx].shape[0]-self.ncond-self.npred)
            for i in range(self.ncond):
                C.append([])
                for x in Box[0][EpisodeIdx][StartStateIdx+i]:
                    C[i].append([x])
                for x in Box[1][EpisodeIdx][StartStateIdx+i]:
                    C[i].append([x])

            for i in range(self.npred):
                B.append([])
                D.append([])
                for k,x in enumerate(Box[0][EpisodeIdx][StartStateIdx+self.ncond+i]):
                    B[i].append([x-C[i][k][0]])
                    D[i].append([x])
            return C,B,D
        else:
            CC=[]
            BB=[]
            DD=[]
            for EpisodeIdx in range(start,final):
                C=[]
                B=[]
                D=[]
                #Box has shape (obs-act,episode ,state|action idx,state|action component)
                #NN  Input : (batch, ncond,state+action,1)
                #    return : (batch,npred,state,1)
                StartStateIdx = RI(0,Box[0][EpisodeIdx].shape[0]-self.ncond-self.npred)
                for i in range(self.ncond):
                    C.append([])
                    for x in Box[0][EpisodeIdx][StartStateIdx+i]:
                        C[i].append([x])
                    for x in Box[1][EpisodeIdx][StartStateIdx+i]:
                        C[i].append([x])

                for i in range(self.npred):
                    B.append([])
                    D.append([])
                    for k,x in enumerate(Box[0][EpisodeIdx][StartStateIdx+self.ncond+i]):
                        B[i].append([x-C[i][k][0]])
                        D[i].append([x])
                CC.append(C)
                BB.append(B)
                DD.append(D)
            return CC,BB,DD

        raise ValueError("No other choice except Random. Sorry~~")


    def get_batch(self,Type):
        condres=[]
        respredres=[]
        predres=[]
        dtype=tf.float32
        if Type=="test" or Type=="valid":
            condres,respredres,predres=self.get_one(Type)
            return tf.convert_to_tensor(condres,dtype),tf.convert_to_tensor(respredres,dtype),tf.convert_to_tensor(predres,dtype)
        else:
            for i in range(self.BatchSize):
                cond , restar,tar= self.get_one(Type)
                condres.append(cond)
                respredres.append(restar)
                predres.append(tar)
            return tf.convert_to_tensor(condres,dtype),tf.convert_to_tensor(respredres,dtype),tf.convert_to_tensor(predres,dtype)



if __name__ =="__main__":
    tf.enable_eager_execution()
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,default="./data/p_samples",help="where all saved samples are")
    parser.add_argument("--npred",type=int,default=1)
    parser.add_argument("--ncond",type=int,default=1)
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--valid",help="Do cross validation. Default : True",default="True")
    parser.add_argument("--n_trials",type=int,default=1000)
    parser.add_argument("--env_id",type=str,default="Acrobot-v1")

    par =parser.parse_args()
    image_loader=DataMaker(par)
