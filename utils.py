from os import path
import pickle
import Config

def CheckModel(mode_name,train_func,continue_train = False,retrain = False):
    f_path = "%s/ModelPickle/%s.pickle"%(Config.envir_path,mode_name)
    model = None
    if path.exists(f_path) and not retrain:
        print("[npc report] model:%s ,file exists,loading"%(mode_name),end = '')
        with open(f_path,"rb+") as model:
            model = pickle.load(model)
            print("... loaded")
            if continue_train:
                print("[npc report] model:%s ,file exists,user choose to continue train the model"%(mode_name))
                model = train_func(model)
                SaveObj(model,mode_name)
    else:
        print("[npc report] model:%s ,file not exists,try to train the model"%(mode_name))
        model = train_func(None)
        SaveObj(model,mode_name)
    return model


def SaveObj(obj,name = "obj",path = Config.envir_path):
    import pickle
    f_name = "%s/ModelPickle/%s.pickle"%(path,name)
    with open(f_name, 'wb+') as net_file:
        pickle.dump(obj,net_file)

def Counter(lst,res = {True:0,False:0}):
    for l in lst:
        if l in res:
            res[l] += 1
        else:
            res[l] = 1
    return res

if __name__ == "__main__":
    print(Counter([1,1,1,1,2,3,3,1]))