import torch
import torch.nn as nn
import Config


# 定义网络结构
class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,1,1,padding=0),
            nn.MaxPool2d(3),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1,1,2,padding=0),
            nn.MaxPool2d(3),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1,1,3,padding=0),
            nn.MaxPool2d(3),
            nn.ReLU()
        )
        self.mlp_hidden1 = nn.Linear(4096,4096)
        self.mlp = nn.Linear(4096,2)

    def forward(self, x):
        x1,x2,x3 = self.conv1(x),self.conv2(x),self.conv3(x)
        #print(x1.size(),x2.size(),x3.size())
        x1,x2,x3 = torch.flatten(x1),torch.flatten(x2),torch.flatten(x3)
        padding = 4096 - (x1.shape[0] + x2.shape[0] + x3.shape[0])
        if padding > 0:
            catx = torch.cat([x1,x2,x3,torch.zeros(padding).to(Config.device)],dim=0)
        else:
            catx = torch.cat([x1,x2,x3],dim=0)[:4096]
        # print(catx.shape)

        x = self.mlp_hidden1(catx)
        x = torch.relu(x)
        x = self.mlp(x)
        x = torch.softmax(x,dim=0)
        return x

    def save(self,info = ""):
        import pickle
        f_name = "%s/ModelPickle/mscnn%s.pickle"%(Config.envir_path,info)
        self.to(torch.device("cpu"))
        with open(f_name, 'wb+') as net_file:
            pickle.dump(self,net_file)
        self.to(self.device)

    def Test(net,dataset):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print("[npc report] test: your device is ",device)
        net.device = device
        net.to(device)

        acu = 0
        # last_loss,count_last_loss = 10,0
        for x,y in dataset:
            x,y = x.unsqueeze(1).to(device),y.to(device)
            o = net(x)
            if torch.argmax(o) == torch.argmax(y): 
                acu += 1
            
        return 100 * acu / len(dataset)

    def Train(net,dataset,testdataset = None):
        from torch.utils.data import DataLoader
        # loader = DataLoader(dataset, batch_size = Config.batch_size, shuffle=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("[npc report] your device is ",device)
        net.device = device
        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
        loss_func = torch.nn.BCELoss()
        mean_losses = []

        # last_loss,count_last_loss = 10,0
        for t in range(4):
            t_losses = []
            count = 0
            acu = 0


            for x,y in dataset:
                x,y = x.unsqueeze(1).to(device),y.to(device)
                o = net(x)
                if torch.argmax(o) == torch.argmax(y): 
                    acu += 1
                # print(o,y)
                loss = loss_func(o,y)
                
                t_losses.append(loss.tolist())

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
                
                mean_losses.append(sum(t_losses)/len(t_losses))
                count += 1
                if count%Config.batch_size == 0:
                    print("[npc report]","echo:",t,
                    "(%d/%d[%2.2f%%])"%(count,len(dataset),100*count/len(dataset)),
                        "loss:",mean_losses[-1],
                        "train_acu:",100*acu/count,
                        "test_acu","None" if testdataset == None else net.Test(testdataset)
                        )

            net.save()

        return net