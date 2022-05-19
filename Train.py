import utils
import sys
sys.path.append("./")
import Config

from Models.CNN import CNNnet
from DatasetAPI.AmazonReviews import CsvDataset as amazon

#第零步，把数据读进内存

import os

# dtls = []
# for csvs in os.listdir(Config.data_path):
#     dtls.append(amazon("%s/%s"%(Config.data_path,csvs)))

train_dataset,test_dataset = amazon("%s/%s"%(Config.data_path,"video.csv")).Split()

print("[npc report] your data is train:test = ",len(train_dataset),":",len(test_dataset))

mscnn = utils.CheckModel(
    'or_mscnn',
    lambda nl:CNNnet(),
    continue_train=False,
    retrain=False
)

mscnn.name="video"
topic_tokenizer = utils.CheckModel(
    'mscnn_video',
    lambda nl:mscnn.Train(train_dataset,test_dataset),
    continue_train=True,
    retrain=False
)
