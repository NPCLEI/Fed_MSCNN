class Fed_Clients:
    def __init__(self,dataset,model):
        self.dataset = dataset
        self.model = model

    def Train(self):
        self.model.train(self.dataset)

    def Upgrade(self,server_model):
        self.model = server_model

    def GetParam(self):
        return self.model

class Fed_Server:
    def __init__(self,model,clients):
        self.model = model
        self.clients = clients

    def Train(self):
        for c in self.clients:
            c.Train()

    def UpgradeParam(self):
        pass