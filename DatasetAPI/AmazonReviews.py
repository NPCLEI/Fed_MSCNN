import torch
from torch.utils.data import Dataset as torchDataset
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

class CsvDataset(torchDataset):
    def __init__(self,csv_path,read_columns = True,sp = ','):
        super(CsvDataset, self).__init__()
        print("[npc report]read data...",end = '')
        self.columns = None
        self.data = {}
        with open(csv_path,"r",encoding="utf8") as csv:
            if read_columns:
                rc = CsvDataset.__splitLine__(csv.readline())
                self.columns = dict([(rc[i],i) for i in range(len(rc))])
            while csv.readable():
                line = csv.readline()
                self.data.append(CsvDataset.__splitLine__(line))
        print("done")

    def __splitLine__(line,sp = ','):
        exist_quotation = False
        ll = len(line)
        lst = 0
        res = []
        for i in range(1,ll):
            if line[i] == "," and not exist_quotation:
                res.append(line[lst:i])
                lst = i + 1
            elif line[i] == '"':
                exist_quotation = not exist_quotation
        res.append(line[lst:])
        return res

    def SplitData(self,cls = []):
        pass

    def ShowLabels(self):
        print()

class PdCsvDataset(torchDataset):
    def __init__(self,csv_path = "../dataset.csv"):
        super(PdCsvDataset, self).__init__()
        self.csv = pd.read_csv(csv_path)
        self.dataidx = []
        self.labels = []
        self.statistics()

    def statistics(self,read_num = 1000,cls = "reviews.doRecommend"):
        tabu = {}
        for idx,cls in enumerate(self.csv[cls]):
            if type(cls) != type(True):
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
        return embedding,torch.Tensor([tl,1-tl])

    def __len__(self) -> int:
        return len(self.dataidx)


    def Split(self,testPor = 0.3):
        import copy
        train = copy.copy(self)
        spl = int(len(self)* 0.5 * testPor)
        train.labels = self.labels[:spl] + self.labels[int(len(self)* 0.5):spl]
        train.dataidx = self.dataidx[:spl] + self.dataidx[int(len(self)* 0.5):spl]

if __name__ == "__main__":
    count = 0
    pcsv = PdCsvDataset()
    print(len(pcsv))

    
    # txt,cls = pcsv[0]
    # inputs = tokenizer("I love you.", return_tensors='pt')
    # with torch.no_grad():
    #     all_encoder_layers = model(**inputs)
    #     print(all_encoder_layers.last_hidden_state)
    for txt,cls in pcsv:
        print(txt.shape,cls)
        if count > 20:
            break
        count += 1