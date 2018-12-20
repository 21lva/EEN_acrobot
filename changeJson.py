import json

A={}
A["nInHidden"]=100
A["nTransporter"]=2
A["npred"]=8
A["ncond"]=8
A["nState"]=6
A["nAction"]=1
A["n_latent"]=16
A["nfeature"]=20
A["batch_size"]=64
A["lr"]=0.001
A["nEpoch"]=1000
A["epoch_size"]=20
A["loss"]="l2"
A["valid"]="True"
A["ScalerType"]="MM"

print(A)
f=open("config.json","w",encoding="utf-8")
json.dump(A,f,indent="\t")
f.close()
