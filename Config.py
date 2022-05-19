import platform

envir_system = platform.system()
print("[npc report] your system is ",envir_system)

batch_size = 300
read_data_num = 100

if envir_system == "Windows":
    envir_path = "."
    data_path = "D:/Dataset/NLP-SA-Sentence-Level/Amazon"
else:
    envir_path = "/root/autodl-tmp/Fed_MSCNN"
    data_path = "/root/autodl-tmp/Dataset"
    batch_size = 300




import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")



vocLen = 10000

TopicModel_train_dataset = ""
