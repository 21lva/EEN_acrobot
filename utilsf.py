import json,os
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,default="Baseline")
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--lr",type=float,default=0.005)
parser.add_argument("--nEpoch",type=int,default=100)
parser.add_argument("--epoch_size",type=int,default=5)
parser.add_argument("--loss",type=str,default="l2",help="l1 or l2")

parser.add_argument("--nfeature",type=int,default=64)
parser.add_argument("--nState",type=int,default=15)
parser.add_argument("--nAction",type=int,default=6)
parser.add_argument("--npred",type=int,default=20)
parser.add_argument("--ncond",type=int,default=1)
parser.add_argument("--n_latent",type=int,default=10)
parser.add_argument("--nInHidden",type=int,default=1500)
parser.add_argument("--valid",type=str,default="True")


par = parser.parse_args()

def read_data(filename):
    f=open(filename,"r")
    return json.load(f)

def json_to_par(par,filename="config.json"):
    JsonObj=read_data(filename)
    for key in JsonObj.keys():
        par.__setattr__(key,JsonObj[key])



def log(logfile,logtext):
    if not os.path.isdir(os.path.dirname(logfile)):
        os.system("mkdir -p "+os.path.dirname(logfile))
    f = open(logfile,"a")
    f.write(str(datetime.now())+logtext+"\n")
    f.close()
    print(logtext)


if __name__=="__main__":
    print(dir(par))
    print(par.__getattribute__("ncond"))
    print(par.ncond)
    json_to_par(par)
    print(par.ncond)
