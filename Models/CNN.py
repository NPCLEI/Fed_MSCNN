import torch
import torch.nn as nn
import Config
from torch.utils.data import DataLoader
import utils
import math

# 定义网络结构
class CNNnet(nn.Module):
    def __init__(self,name = ''):
        super(CNNnet,self).__init__()
        self.name = name
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,1,(1,768),padding=0),
            nn.MaxPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout2d(0.1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1,1,(2,768),padding=0),
            nn.MaxPool2d((2,1)),
            nn.Flatten(),
            nn.Dropout2d(0.1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1,1,(3,768),padding=0),
            nn.MaxPool2d((3,1)),
            nn.Flatten(),
            nn.Dropout2d(0.1),
            nn.ReLU()
        )

        self.mlpInLen = 937
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.mlpInLen),
            nn.Linear(self.mlpInLen,512),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Linear(512,2)
        )

    def forward(self, x):

        x1,x2,x3 = self.conv1(x),self.conv2(x),self.conv3(x)
        catx = torch.cat([x1,x2,x3],dim=1)
        x = self.mlp(catx)
        x = torch.softmax(x,dim=0)
        return x

    def save(self,info = ""):
        import pickle
        f_name = "%s/ModelPickle/mscnn_%s.pickle"%(Config.envir_path,info)
        self.to(torch.device("cpu"))
        with open(f_name, 'wb+') as net_file:
            pickle.dump(self,net_file)
        self.to(self.device)

    def Test(net,dataset):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print("[npc report] test: your device is ",device)
        net.device = device
        net.to(device)
        loader = DataLoader(dataset, batch_size = Config.batch_size, shuffle=True,collate_fn=dataset.collate_func)
        loss_func = torch.nn.BCELoss()

        acu = 0
        counter = {}
        batch_count = 0
        # last_loss,count_last_loss = 10,0
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            o = net(x)
            # print(y)
            loss = loss_func(o,y)
            cmp = o.argmax(1).eq(y.argmax(1))
            acu = utils.Counter(cmp.tolist(),counter)
            batch_count += 1

            print("[npc report]","testing : echo:",-1,
                "(%d/%d[%2.2f%%])"%(
                        batch_count,
                        len(dataset)/Config.batch_size,
                        100*batch_count/(len(dataset)/Config.batch_size)
                    )
                )
        
        acuv = 100*acu[True]/(acu[True]+acu[False])
        print("[npc report] test: acu_obj:",acu,"acu :",acuv)
        return acuv

    def Train(net,dataset,testdataset = None):
        loader = DataLoader(dataset, batch_size = Config.batch_size, shuffle=True,collate_fn=dataset.collate_func)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("[npc report] your device is ",device)
        net.device = device
        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001 ,weight_decay = 1e-3)
        loss_func = torch.nn.BCELoss()
        mean_losses = []
        prgl = math.ceil(len(dataset)/Config.batch_size)

        # last_loss,count_last_loss = 10,0
        for t in range(100):
            t_losses = []

            batch_count = 0
            for x,y in loader:
                x,y = x.to(device),y.to(device)

                o = net(x)
                #计算准确率
                cmp = o.argmax(1).eq(y.argmax(1))
                acu = utils.Counter(cmp.tolist(),{True:0,False:0})

                loss = loss_func(o,y)    
                
                t_losses.append(loss.tolist())

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
                
                mean_losses.append(sum(t_losses)/len(t_losses))
                batch_count += 1
                print("[npc report]","echo:",t,
                    "(%d/%d[%2.2f%%])"%(
                            batch_count,
                            prgl,
                            100*batch_count/prgl
                        ),
                        "loss:",mean_losses[-1],
                        "batch acu:",acu,
                        "acuv:",100*acu[True]/(acu[True]+acu[False])
                    )
            if mean_losses[-1] < 0.001:
                break
            T = -1
            if testdataset != None:
                T = net.Test(testdataset)
            net.save("%s[T%2.3f]"%(net.name,T))

        return net