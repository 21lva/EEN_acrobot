#import nmodels as models
import matplotlib.pyplot as plt
import models as models
import utilsf as utils#,parsing
import pa as parsing
import argparse,os
import tensorflow as tf
import numpy as np
import csv



#parsing procedure
parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,default="Baseline")
parser.add_argument("--env_id",type=str,default="Acrobot-v1")
parser.add_argument("--n_trials",type=int,default=1000)
parser.add_argument("--train_size",type=int,default=700)
parser.add_argument("--test_size",type=int,default=100)
parser.add_argument("--ScalerType",type=str,default="StandardScaler")
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--lr",type=float,default=0.005)
parser.add_argument("--nEpoch",type=int,default=1000)
parser.add_argument("--epoch_size",type=int,default=5)
parser.add_argument("--loss",type=str,default="l2",help="l1 or l2")

parser.add_argument("--nfeature",type=int,default=64)
parser.add_argument("--nState",type=int,default=15)
parser.add_argument("--nAction",type=int,default=6)
parser.add_argument("--npred",type=int,default=20)
parser.add_argument("--ncond",type=int,default=100)
parser.add_argument("--n_latent",type=int,default=10)
parser.add_argument("--nInHidden",type=int,default=1500)
parser.add_argument("--valid",type=str,default="True")

parser.add_argument("--datapath",type=str,default ="./data/",help="root directory where all the data will save")
parser.add_argument('--save_dir', type=str,
                     default='./results/')

"""
Saving Model : ./data/[model name]/model/
Saving Log : ./data/[model name]/log/
"""

par = parser.parse_args()


"""
Each model in json file has
    height,width,
    nc(#channels of images),n_actions,
    ncond(#input images),npred(#output images),
    phi_fc_size,dataloader,
    datapath
"""

par.datapath = par.datapath+par.model+"/"
utils.json_to_par(par)

par.filename = "model:{}-lr:{}-epoch_size:{}-nEpoch:{}-batch_size:{}-nfeature:{}-loss:{}-nInhidden:{}-nlatent:{}".format(
        par.model,par.lr,par.epoch_size,par.nEpoch,par.batch_size,par.nfeature,par.loss,par.nInHidden,par.n_latent
        )

#data parsing
dataloader =parsing.DataMaker(par)

#train , validation
#A train epoch
def train_epoch(model,optimizer):
    total_loss=0.0
    for _ in range(par.epoch_size):
        Input,Target,_= dataloader.get_batch("train")
        gradient,loss = model.compute_gradients(Input,Target)
        model.apply_gradients(gradient)
        loss = tf.keras.backend.mean(loss)
        #print(loss)
        total_loss += loss
    return total_loss/par.epoch_size


def validation_epoch(model):
    total_loss =0.0
    for _ in range(par.epoch_size):
        Input,Target,_= dataloader.get_batch("valid")
        gradient,loss = model.compute_gradients(Input,Target)
        loss = tf.keras.backend.mean(loss)
        total_loss += loss
    return total_loss/par.epoch_size

def train(model,optimizer):
    #prepare for saving
    os.system("mkdir -p "+par.datapath)
    os.system("mkdir -p "+par.datapath+"model")
    os.system("mkdir -p "+par.datapath+"log")
    os.system("mkdir -p "+par.datapath+"pic")

    train_losses,valid_losses =[],[]
    best_valid_loss =1000000.0
    f=open(par.datapath+"log/"+par.filename+".csv","w")
    wr=csv.writer(f)
    for i in range(par.nEpoch):
        train_losses.append(train_epoch(model,optimizer))
        valid_losses.append(validation_epoch(model))
        wr.writerow([train_losses[-1],valid_losses[-1]])
        if valid_losses[-1] < best_valid_loss:
            best_valid_loss = valid_losses[-1]
            #save model
            model.DEncoder.save_weights(par.datapath+"model/"+par.filename+".Emodel")
            model.DDecoder.save_weights(par.datapath+"model/"+par.filename+".Dmodel")
        #print log
        logtxt = "model:{} / epoch:{} / train loss:{} / validation loss:{} / best validation loss:{}".format(par.model,i,train_losses[-1],valid_losses[-1],best_valid_loss)
        utils.log(par.datapath+"log/"+par.filename+".txt",logtxt)
    f.close()
    plt.figure()
    plt.plot(train_losses,color="blue")
    plt.plot(valid_losses,color="green")
    plt.title("Train(blue) vs Valid(green) of {}".format(par.loss))
    plt.savefig(par.datapath+"pic/"+par.filename+".png")
    plt.clf()

if __name__=="__main__":
    #optimizer =tf.train.GradientDescentOptimizer(par.lr)
    #optimizer =tf.train.RMSPropOptimizer(par.lr)
    optimizer =tf.train.AdamOptimizer(par.lr)
    model =models.DeterministicNetwork(optimizer,par)
    try:
        model.DEncoder.load_weights("./data"+"/Baseline/model/"+par.filename+".Emodel")
        model.DDecoder.load_weights("./data"+"/Baseline/model/"+par.filename+".Dmodel")
        print("************Using saved Model")
    except:
        print("************No available saved Model")
    print("==================Lets start Train=============")
    print("=================File Name=====================")
    print(par.filename)
    print("===============================================")
    train(model,optimizer)
