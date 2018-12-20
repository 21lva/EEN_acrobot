### EEN_acrobot


Error encoding network for acrobot
error encoding network by tensorflow.keras. 
The paper of EEN : https://arxiv.org/abs/1711.04994

need:
python>=3.5
tensorflow (any version that support tf.keras)
numpy
matplotlib
etc....

##Train deterministic model:
    \'python train_d.py\'
    
    
After training deterministic model, a stochastic model should be trained.
    
##Train stochastic model:
    python train_s.py
    
    
##Predict and show image of each state prediction:
    python predict.py
