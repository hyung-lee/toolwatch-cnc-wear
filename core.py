from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
ROOT=Path(__file__).resolve().parent
class WearCNN(nn.Module):
 def __init__(self):
  super().__init__();layers=[];inc=3
  for out in [16,32,64,96]:
   layers += [nn.Conv2d(inc,out,3,padding=1),nn.GroupNorm(4,out),nn.ReLU(),nn.MaxPool2d(2)];inc=out
  self.net=nn.Sequential(*layers,nn.AdaptiveAvgPool2d((3,3)),nn.Flatten(),nn.Linear(864,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
 def forward(self,x):return self.net(x).flatten()
def tensor_image(im):
 a=np.array(im.convert('RGB').resize((128,128)),dtype=np.float32)/255
 return torch.from_numpy(a.transpose(2,0,1).copy())
def metrics(y,p):
 y=np.asarray(y);p=np.asarray(p);den=((y-y.mean())**2).sum()
 return dict(MAE=float(np.abs(y-p).mean()),RMSE=float(np.sqrt(((y-p)**2).mean())),R2=float(1-((y-p)**2).sum()/den) if den>0 else None,n=len(y))
def history_features(t,w):
 t=np.asarray(t,dtype=float);w=np.asarray(w,dtype=float)
 if len(t)<3 or np.any(np.diff(t)<=0):raise ValueError('서로 다른 증가하는 가공 회차 3개 이상 필요')
 slope=float(np.polyfit(t[-5:],w[-5:],1)[0])
 return np.array([w[-1],slope,t[-1],w[-min(5,len(w)):].mean(),w[-1]-w[0]],dtype=np.float32)
def load_model():
 ck=torch.load(ROOT/'wear.pt',map_location='cpu',weights_only=True)
 m=WearCNN();m.load_state_dict(ck['state']);m.eval();return m,ck
@torch.inference_mode()
def predict(m,ck,im):return max(0.,float(m(tensor_image(im)[None])[0])*ck['scale']+ck['mean'])
