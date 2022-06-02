from tqdm import tqdm
import os

import torch


from MVRA.dataset import ScannetDataset
from MVRA.Model import Multi_view_reconstruction
from MVRA.config import get_config
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

def Evaluate(model,evaluate_DataLoader):
    
  
    pbar = tqdm(evaluate_DataLoader)
    acc_loss=[]
    model.eval()
    for inputs in pbar:
        for k,v in inputs.items():
            if(k == 'tsdf_list'):
              for i in range(3):
                inputs[k][i]=inputs[k][i].to(torch.device('cuda'))
            else:
              inputs[k] = inputs[k].to(torch.device('cuda'))

        with torch.no_grad():
          loss = model(inputs)
        
        acc_loss .append(loss.item())
        pbar.set_description('  loss  '+str(sum(acc_loss)/len(acc_loss)))
      
    avg=sum(acc_loss)/len(acc_loss)
    return avg



def train(model,train_DataLoader,save_path=False,epochs=40,eval_data=False):
    
    model = model.to(torch.device('cuda'))
    optimizer =  torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0)

    if(save_path and os.path.exists(save_path)):
      ckpt = torch.load(save_path)
      model.load_state_dict(ckpt['model'])
      optimizer.load_state_dict(ckpt['optimizer'])

    total_loss = []
    bestloss=1000

    for epc in range(epochs):
        pbar = tqdm(train_DataLoader)
        acc_loss=[]
        
        model.train()
        for inputs in pbar:
            
            for k,v in inputs.items():
                if(k == 'tsdf_list'):
                  for i in range(3):
                    inputs[k][i]=inputs[k][i].to(torch.device('cuda'))
                else:
                  inputs[k] = inputs[k].to(torch.device('cuda'))

            
            optimizer.zero_grad()
            loss = model(inputs)
            
            acc_loss .append(loss.item())
            total_loss.append(loss.item())
            

            
            loss.backward()
            optimizer.step()
            pbar.set_description('epoch  :  ' +str(epc)+'  loss  '+str(sum(acc_loss)/len(acc_loss)))
        avg=sum(acc_loss)/len(acc_loss)
        file1 = open("/root/Projectwork2/metrics/finalnew.txt", "a")
        for ls in acc_loss:
          file1.write(str(ls)+'\n')
        if epc%4 == 0 and eval_data:
           eval_Avg = Evaluate(model,eval_data)
           print("Evaluation result loss ",eval_Avg)

        file1.close()
        
        if(avg<bestloss and save_path):
          torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, save_path)
          bestloss=avg





path = "/root/Projectwork2/Assets/finalmodel.pth"
cfg = get_config()
model=Multi_view_reconstruction(cfg,Attention=True)
traindata=ScannetDataset("./../Datasets/scans96_train/" , "./../Datasets/tsdf96_train/",cfg,augment=False)
traindataflow = torch.utils.data.DataLoader(traindata,batch_size=2,shuffle=True)

valdata=ScannetDataset("./../Datasets/scans96_val/" , "./../Datasets/tsdf96_val/",cfg)
valdataflow = torch.utils.data.DataLoader(valdata,batch_size=2,shuffle=False)
train(model,traindataflow,path,eval_data=False)


valdata=ScannetDataset("./../Datasets/scans_val/" , "./../Datasets/tsdf_val/",cfg)
valdataflow = torch.utils.data.DataLoader(valdata,batch_size=1,shuffle=False)
model.load_state_dict(torch.load(path)['model'])
model = model.to(torch.device('cuda'))
Evaluate(model,valdataflow)






