from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset as torchDataset
import pandas as pd
from transformers import BertTokenizer, BertModel
import Config

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(Config.device)

class CsvDataset(torchDataset):
    def __init__(self,csv_path,read_columns = True,sp = ','):
        super(CsvDataset, self).__init__()
        self.dataTell = []
        if csv_path == None:
            return
        print("[npc report]read data...",end = '')
        self.raw_lines = open(csv_path,"r",encoding="utf8").read().split('\n')
        print("done.")
        
    def __splitLine__(line,sp = ','):
        exist_quotation = False
        ll = len(line)
        lst = 0
        res = []
        for i in range(1,ll):
            if line[i] == sp and not exist_quotation:
                res.append(line[lst:i])
                lst = i + 1
            elif line[i] == '"':
                exist_quotation = not exist_quotation
        res.append(line[lst:])
        return res

    def __getitem__(self, idx: int):
        try:
            label,txt = CsvDataset.__splitLine__(self.raw_lines[idx])
            tl = 1 if label == "True" else 0
            inputs = tokenizer(txt ,padding = 'max_length', return_tensors='pt',truncation  = True).to(Config.device)

            with torch.no_grad():
                embedding = model(**inputs).last_hidden_state

            return embedding,torch.Tensor([[tl,1-tl]]).to(Config.device)
            
        except Exception as e:
            print("[npc report] Unhandle Eorr:",e,"auto handle:","skip")
            return -1,-1

    def __len__(self) -> int:
        return len(self.raw_lines)

    def collate_func(self,batch_dic):
        xs,ys = [],[]
        for x,y in batch_dic:
            if type(x) == type(-1):
                continue
            xs.append(x.unsqueeze(0))
            ys.append(y)
        return torch.cat(xs,dim=0),torch.cat(ys,dim=0)

    def Split(self,testPor = 0.3):
        train,test = CsvDataset(None),CsvDataset(None)
        half = int(len(self)* 0.5) # 1000
        spl = int(half * (1 -testPor)) # 700 
        train.raw_lines = self.raw_lines[:spl] + self.raw_lines[half:half + spl]

        test.raw_lines = self.raw_lines[spl:half] + self.raw_lines[-(half - spl):]

        return train,test

    def Merge(dtls:list):
        res = CsvDataset(None)
        res.raw_lines = []
        for dt in dtls:
            res.raw_lines += dt.raw_lines
        return res

class PdCsvDataset(torchDataset):
    def __init__(self,csv_path = None):
        super(PdCsvDataset, self).__init__()
        self.dataidx = []
        self.labels = []
        if csv_path != None:
            self.csv = pd.read_csv(csv_path,sep=',')
            self.statistics()

    def collate_func(self,batch_dic):
        xs,ys = [],[]
        for x,y in batch_dic:
            xs.append(x.unsqueeze(0).unsqueeze(0))
            ys.append(y.unsqueeze(0))
        return torch.cat(xs,dim=0),torch.cat(ys,dim=0)


    def statistics(self,read_num = 1000,cls = "reviews.doRecommend"):
        tabu = {}
        for idx,cls in enumerate(self.csv[cls]):
            if type(cls) != type(True):
                continue
            print(self.csv.iloc[idx]["reviews.text"])
            inputs = tokenizer(self.csv.iloc[idx]["reviews.text"], return_tensors='pt')
            if len(inputs) > 47:
                continue
            if cls in tabu.keys():
                if tabu[cls] <= read_num:
                    self.dataidx.append(idx)
                    self.labels.append(cls)
                    tabu[cls] += 1
            else:
                tabu.setdefault(cls,1)
                self.labels.append(cls)
        self.clses = list(tabu.keys())
        self.tabu = tabu
        print("[npc report] PdCsvDataset stat :",tabu)


    def __getitem__(self, index: int):
        idx = self.dataidx[index]
        tl = 1 if self.labels[index] else 0
        inputs = tokenizer(self.csv.iloc[idx]["reviews.text"], return_tensors='pt')
        with torch.no_grad():
            embedding = model(**inputs).last_hidden_state
        
        c,h,w = embedding.shape
        if  h < 46:
            embedding = torch.cat([embedding,torch.zeros(1,46- h,w)],dim = 1)
        return embedding[0,:46,:],torch.Tensor([tl,1-tl])

    def __len__(self) -> int:
        return len(self.dataidx)


    def Split(self,testPor = 0.3):
        train,test = PdCsvDataset(None),PdCsvDataset(None)
        half = int(len(self)* 0.5) # 1000
        spl = int(half * (1 -testPor)) # 700 
        train.labels = self.labels[:spl] + self.labels[half:half + spl]
        train.dataidx = self.dataidx[:spl] + self.dataidx[half:half + spl]
        train.csv = self.csv

        test.labels = self.labels[spl:half] + self.labels[-(half - spl):]
        test.dataidx = self.dataidx[spl:half] + self.dataidx[-(half - spl):]
        test.csv = self.csv

        return train,test

if __name__ == "__main__":
    count = 0
    pcsv = PdCsvDataset("../dataset.csv")
    print(len(pcsv))
   
    # txt,cls = pcsv[0]
    # inputs = tokenizer("It is the time you have wasted for your rose that makes your rose so important.", return_tensors='pt')
    # with torch.no_grad():
    #     all_encoder_layers = model(**inputs)
    #     print(all_encoder_layers.last_hidden_state)

    plty1 = []
    plty2 = []
    x = []
    count = 0
    for txt,cls in pcsv:
        x.append(count)
        count+=1
        plty1.append(txt.shape[1])
        plty2.append(txt.shape[2])

    print(sum(plty1)/len(plty1),sum(plty2)/len(plty2))
    plt.plot(x,plty1,color = 'red')        
    plt.plot(x,plty2,color = 'green')
    plt.show()