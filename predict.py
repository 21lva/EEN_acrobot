import tensorflow as tf
import numpy as np
import argparse,os,glob,csv
#import nmodels as models
import models
import pa as parsing
import matplotlib.pyplot as plt
from random import randint as ri
import changeJson
import utilsf as utils

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,default="Latent")
parser.add_argument("--env_id",type=str,default="Acrobot-v1")
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--lr",type=float,default=0.0005)
parser.add_argument("--nEpoch",type=int,default=100)
parser.add_argument("--epoch_size",type=int,default=5)
parser.add_argument("--loss",type=str,default="l2",help="l1 or l2")
parser.add_argument("--n_trials",type=int,default=1000)
parser.add_argument("--train_size",type=int,default=700)
parser.add_argument("--test_size",type=int,default=100)
parser.add_argument("--ScalerType",type=str,default="StandardScaler")

parser.add_argument("--nfeature",type=int,default=64)
parser.add_argument("--nState",type=int,default=15)
parser.add_argument("--nAction",type=int,default=6)
parser.add_argument("--npred",type=int,default=20)
parser.add_argument("--ncond",type=int,default=100)
parser.add_argument("--n_latent",type=int,default=4)
parser.add_argument("--nInHidden",type=int,default=1000)
parser.add_argument("--valid",type=str,default="True")

parser.add_argument("--tfilename",type=str,default="1.csv")
parser.add_argument("--datapath",type=str,default ="./data/",help="root directory where all the data will save")
parser.add_argument('--save_dir', type=str,
                     default='./results/')

par= parser.parse_args()
utils.json_to_par(par)
par.data_dir = par.datapath+"p_samples"
par.datapath = par.datapath+par.model+"/"
par.filename = "model:{}-lr:{}-epoch_size:{}-nEpoch:{}-batch_size:{}-nfeature:{}-loss:{}-nInhidden:{}-nlatent:{}".format(
        par.model,par.lr,par.epoch_size,par.nEpoch,par.batch_size,par.nfeature,par.loss,par.nInHidden,par.n_latent
        )

dataloader = parsing.DataMaker(par)

def load_model(ModelNames,par,IsBaseline=True):
    #return model
    optimizer = tf.train.AdamOptimizer(par.lr)
    if IsBaseline:
        model=models.DeterministicNetwork(optimizer,par)
        model.DEncoder.load_weights(ModelNames[0])
        model.DDecoder.load_weights(ModelNames[1])
    else:
        model=models.LatentNetwork(optimizer,par)
        model.load_weights(ModelNames)
    return model

def get_cond(dataloader,Type,IsRandom=False):
    if IsRandom:
        for i in range(ri(1,100)):
            Cond,Target ,_= dataloader.get_batch("valid")
    return dataloader.get_batch(Type)

def DrawGraph(par,dataloader,save_dir,IsBaseline=True,nWant=1):
    NumCS = par.nState+par.nAction
    NumPS = par.nState

    os.system("mkdir -p "+save_dir+"/"+par.model)
    graph_dir = save_dir+"/"+par.model+"/"
    if IsBaseline:
        fp = open(save_dir+"/PredUsing_"+par.tfilename,"r")#i must be par.filename
        ft = open(save_dir+"/TargetUsing_"+par.tfilename,"r")#i must be par.filename

        PR = csv.reader(fp)
        TR = csv.reader(ft)
        for k in range(nWant):
            PX=[[] for _ in range(par.nTransporter)]
            TX=[[] for _ in range(par.nTransporter)]
            for i in range(par.ncond):
                for j in range(par.nTransporter):
                    x= PR.__next__()[:NumPS]
                    lis = [float(y) for y in x]
                    z= TR.__next__()[:NumPS]
                    lis2 = [float(y) for y in z]
                    PX[j].append(lis[:])
                    TX[j].append(lis2[:])


            for i in range(par.npred):
                for j in range(par.nTransporter):
                    x= PR.__next__()[:NumPS]
                    z= TR.__next__()[:NumPS]
                    lis1 = [float(y) for y in x]
                    lis2 = [float(y) for y in z]

                    PX[j].append(lis1[:])
                    TX[j].append(lis2[:])


            os.system("mkdir -p "+graph_dir+str(nWant))

            for i in range(NumPS):
                for k in range(0,1):#par.nTransporter):
                    plt.plot(np.transpose(PX[k])[i],color="blue")
                    plt.plot(np.transpose(TX[k])[i],color="black")
                plt.title("Target(black) vs Predict(blue)")
                plt.ylim(0,1)
                plt.savefig(graph_dir+str(nWant)+"/"+str(i)+".png",dpi=300)
                plt.clf()
        fp.close()
        ft.close()
    else:
        fpd = open(save_dir+"/DPredUsing_"+par.tfilename,"r")#i must be par.filename
        fpp = open(save_dir+"/PPredUsing_"+par.tfilename,"r")#i must be par.filename
        fpr = open(save_dir+"/RPredUsing_"+par.tfilename,"r")#i must be par.filename
        ft = open(save_dir+"/TargetUsing_"+par.tfilename,"r")#i must be par.filename

        PDR = csv.reader(fpd)
        PPR = csv.reader(fpp)
        PRR = csv.reader(fpr)
        TR = csv.reader(ft)

        for k in range(nWant):
            PDX=[]
            PPX=[]
            PRX=[]
            TX=[]
            for i in range(par.ncond):
                x1= PDR.__next__()[:NumPS]
                lis1 = [float(y) for y in x1]
                x2= PPR.__next__()[:NumPS]
                lis2 = [float(y) for y in x2]
                x3= PRR.__next__()[:NumPS]
                lis3 = [float(y) for y in x3]
                x4= TR.__next__()[:NumPS]
                lis4 = [float(y) for y in x4]
                PDX.append(lis1[:])
                PPX.append(lis2[:])
                PRX.append(lis3[:])
                TX.append(lis4[:])


            for i in range(par.npred):
                d= PDR.__next__()[:NumPS]
                p= PPR.__next__()[:NumPS]
                r= PRR.__next__()[:NumPS]
                t= TR.__next__()[:NumPS]
                lisd = [float(y) for y in d]
                lisp = [float(y) for y in p]
                lisr = [float(y) for y in r]
                lis = [float(y) for y in t]

                PDX.append(lisd[:])
                PPX.append(lisp[:])
                PRX.append(lisr[:])
                TX.append(lis[:])

            os.system("mkdir -p "+graph_dir+str(k+1))
            print("save dir : "+graph_dir)

            for i in range(NumPS):
                plt.plot(np.transpose(PDX)[i],color="blue")
                plt.plot(np.transpose(PPX)[i],color="green")
                plt.plot(np.transpose(PRX)[i],color="red")
                plt.plot(np.transpose(TX)[i],color="black")
                plt.title("Target(black) vs Predict(Deterministic-blue | Random-red | UsingLatent-green)")
                #plt.title("Target(black) vs Predict(UsingLatent-green)")
                plt.savefig(graph_dir+str(k+1)+"/"+str(i)+".png",dpi=300)
                plt.clf()
            break
        fpd.close()
        fpp.close()
        fpr.close()
        ft.close()




#############################################################
#BASELINE MODEL
# ==> deprecated
#############################################################

def BaseLineModelPredictor(model,par,dataloader):
    EnMoFileName= par.datapath+"model/"+par.filename+".Emodel"
    DeMoFileName= par.datapath+"model/"+par.filename+".Dmodel"

    save_dir = par.save_dir+os.path.basename(EnMoFileName)[:-7]
    os.system("mkdir -p "+save_dir+"/"+par.model)


    Input,Target = get_cond(dataloader)
    print(tf.keras.backend.mean(model.compute_loss(Input,Target)))
    Pred = model.predict(Input)
    Input=Input.numpy().tolist()
    Target=Target.numpy().tolist()
    Pred=Pred.numpy().tolist()

    fp = open(save_dir+"/PredUsing_"+par.tfilename,"w")#i must be par.filename
    ft = open(save_dir+"/TargetUsing_"+par.tfilename,"w")#i must be par.filename

    PCsvWriter = csv.writer(fp)
    TCsvWriter = csv.writer(ft)
    #CsvWriter.writerow(columns)


    for i in range(6):
        for j in range(par.ncond):
            for k in range(par.nTransporter):
                Input[i][j][k] = [x[0] for x in Input[i][j][k]]
                TCsvWriter.writerow(Input[i][j][k])
                PCsvWriter.writerow(Input[i][j][k])
        for j in range(par.npred):
            for k in range(par.nTransporter):

                Target[i][j][k]= [x[0] for x in Target[i][j][k]]
                Pred[i][j][k] = [x[0] for x in Pred[i][j][k]]
                PCsvWriter.writerow(Pred[i][j][k])
                TCsvWriter.writerow(Target[i][j][k])
    print("=====Complete : making pred target file=======")
    fp.close()
    ft.close()

    DrawGraph(par,dataloader,save_dir,)


############################################################
#LATENT MODEL
###########################################################

def LatentModelPredictor(model,par,dataloader):
    MoFileName= par.datapath+"model/"+par.filename+".model"

    save_dir = par.save_dir+os.path.basename(MoFileName)[:-6]
    os.system("mkdir -p "+save_dir+"/"+par.model)
    #"""

    zlist =[]
    print("sampling z from traing set")
    for k in range(20):
        Input,resTarget,Target = get_cond(dataloader,"test")
        loss , Pred, g_Pred, z = model.compute_loss(Input,resTarget)
        zlist.append(z)

    Input ,resTarget,Target= get_cond(dataloader,"test")
    loss, Pred,g_Pred,z_true = model.compute_loss(Input,resTarget)
    z = zlist[ri(0,len(zlist)-1)]
    #print(z.shape)

    NewPred = model.decode(Input,z)
    Inputa = Input[:,:,:6,:]
    zlist.append(z_true)

    g_Pred= (g_Pred+Inputa).numpy().tolist()
    Pred= (Inputa+Pred).numpy().tolist()
    NewPred= (NewPred+Inputa).numpy().tolist()
    Target= Target.numpy().tolist()
    Input = Inputa.numpy().tolist()

    fpd = open(save_dir+"/DPredUsing_"+par.tfilename,"w")
    fpp = open(save_dir+"/PPredUsing_"+par.tfilename,"w")
    fpr = open(save_dir+"/RPredUsing_"+par.tfilename,"w")
    ft = open(save_dir+"/TargetUsing_"+par.tfilename,"w")

    DCsvWriter = csv.writer(fpd)
    PCsvWriter = csv.writer(fpp)
    RCsvWriter = csv.writer(fpr)
    TCsvWriter = csv.writer(ft)
    ei = ri(0,100)
    for j in range(par.ncond):
        tmp= invScaling(Input[ei][j],dataloader)
        DCsvWriter.writerow(tmp)
        PCsvWriter.writerow(tmp)
        RCsvWriter.writerow(tmp)
        TCsvWriter.writerow(tmp)
    for j in range(par.npred):
        DCsvWriter.writerow(invScaling(g_Pred[ei][j],dataloader))
        PCsvWriter.writerow(invScaling(Pred[ei][j],dataloader))
        RCsvWriter.writerow(invScaling(NewPred[ei][j],dataloader))
        TCsvWriter.writerow(invScaling(Target[ei][j],dataloader))
    fpd.close()
    fpp.close()
    fpr.close()
    ft.close()
    mse_loss = 0
    mse_Newloss =0
    mse_DeLoss = 0
    mape_loss = 0
    mape_Newloss =0
    mape_DeLoss = 0
    mae_loss = 0
    mae_Newloss =0
    mae_DeLoss = 0

    for _ in range(10):
        for k in range(20):
            Input,resTarget,Target = get_cond(dataloader,"test")
            loss , Pred, g_Pred, z = model.compute_loss(Input,resTarget)
            zlist.append(z)

        Input ,resTarget,Target= get_cond(dataloader,"test")
        loss, Pred,g_Pred,z_true = model.compute_loss(Input,resTarget)
        z = zlist[ri(0,len(zlist)-1)]
        #print(z.shape)

        NewPred = model.decode(Input,z)
        Inputa = Input[:,:,:6,:]
        zlist.append(z_true)

        g_Pred= (g_Pred+Inputa).numpy().tolist()
        Pred= (Inputa+Pred).numpy().tolist()
        NewPred= (NewPred+Inputa).numpy().tolist()
        Target= Target.numpy().tolist()
        Input = Inputa.numpy().tolist()


        tmp=[]
        for episode in Target:
            tmp.append([])
            for state in episode:
                tmp[-1].append(invScaling(state,dataloader))
        Target=tf.convert_to_tensor(tmp,tf.float32)
        tmp=[]
        for episode in g_Pred:
            tmp.append([])
            for state in episode:
                tmp[-1].append(invScaling(state,dataloader))
        g_Pred=tf.convert_to_tensor(tmp,tf.float32)
        tmp=[]
        for episode in Pred:
            tmp.append([])
            for state in episode:
                tmp[-1].append(invScaling(state,dataloader))
        Pred=tf.convert_to_tensor(tmp,tf.float32)
        tmp=[]
        for episode in NewPred:
            tmp.append([])
            for state in episode:
                tmp[-1].append(invScaling(state,dataloader))
        NewPred=tf.convert_to_tensor(tmp,tf.float32)


        #l2 error
        tmse_loss = tf.keras.losses.mean_squared_error(Target,Pred)
        tmse_Newloss = tf.keras.losses.mean_squared_error(Target,NewPred)
        tmse_DeLoss = tf.keras.losses.mean_squared_error(Target,g_Pred)

        mse_loss += tf.keras.backend.mean(tmse_loss)
        mse_Newloss += tf.keras.backend.mean(tmse_Newloss)
        mse_DeLoss += tf.keras.backend.mean(tmse_DeLoss)

        #mape
        tmape_loss = mape(Target,Pred)
        tmape_Newloss = mape(Target,NewPred)
        tmape_DeLoss = mape(Target,g_Pred)

        mape_loss += tf.keras.backend.mean(tmape_loss)
        mape_Newloss += tf.keras.backend.mean(tmape_Newloss)
        mape_DeLoss += tf.keras.backend.mean(tmape_DeLoss)


        #l1 error
        tmae_loss = tf.keras.losses.mae(Target,Pred)
        tmae_Newloss = tf.keras.losses.mae(Target,NewPred)
        tmae_DeLoss = tf.keras.losses.mae(Target,g_Pred)


        mae_loss += tf.keras.backend.mean(tmae_loss)
        mae_Newloss += tf.keras.backend.mean(tmae_Newloss)
        mae_DeLoss += tf.keras.backend.mean(tmae_DeLoss)


    floss = open(save_dir+"/loss_"+par.tfilename,"w")
    loss_writer = csv.writer(floss)
    mse_loss /=10
    mse_Newloss /=10
    mse_DeLoss /=10
    mape_loss /=10
    mape_Newloss /=10
    mape_DeLoss /=10
    mae_loss /=10
    mae_Newloss /=10
    mae_DeLoss /=10


    print("L2 - DeLoss : {} | Loss : {}  | RandomLoss : {}".format(mse_DeLoss,mse_loss,mse_Newloss))
    print("L1 - DeLoss : {} | Loss : {}  | RandomLoss : {}".format(mae_DeLoss,mae_loss,mae_Newloss))
    print("MAPE - DeLoss : {} | Loss : {}  | RandomLoss : {}".format(mape_DeLoss,mape_loss,mape_Newloss))
    loss_writer.writerow([mse_DeLoss,mse_loss,mse_Newloss])
    loss_writer.writerow([mae_DeLoss,mae_loss,mae_Newloss])
    loss_writer.writerow([mape_DeLoss,mape_loss,mape_Newloss])
    floss.close()

    DrawGraph(par,dataloader,save_dir,IsBaseline=False,nWant=6)
def invScaling(state,dataloader):
    return dataloader.StateScaler.inverse_transform(np.array([[x[0] for x in state]]))[0]
def mape(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    diff          = tf.norm(y_true-y_pred, axis=1)
    y_true_length = tf.norm(y_true, axis=1)
    return tf.keras.backend.mean((diff/y_true_length)*100)




############################################################
#DO
###########################################################
if __name__=="__main__":
    if par.model!="Latent":
        EnMoFileName= par.datapath+"model/"+par.filename+".Emodel"
        DeMoFileName= par.datapath+"model/"+par.filename+".Dmodel"
        model = load_model([EnMoFileName,DeMoFileName],par)
        BaseLineModelPredictor(model,par,dataloader)
    else:
        MoFileName= par.datapath+"model/"+par.filename+".model"
        model = load_model(MoFileName,par,IsBaseline=False)
        LatentModelPredictor(model,par,dataloader)
