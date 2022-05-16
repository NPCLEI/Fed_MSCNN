from regex import F
import utils
import sys
sys.path.append("./")
import Config

from Models.CNN import CNNnet
from DatasetAPI.AmazonReviews import PdCsvDataset as amazon

#第零步，把数据读进内存

dataset = amazon("./dataset.csv") 
mscnn = CNNnet()

topic_tokenizer = utils.CheckModel(
    'mscnn',
    lambda nl:mscnn.Train(dataset),
    continue_train=True,
    retrain=False
)
