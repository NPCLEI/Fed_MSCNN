from regex import F
import utils
import sys
sys.path.append("./")
import Config

from Models.CNN import CNNnet
from DatasetAPI.AmazonReviews import PdCsvDataset as amazon

#第零步，把数据读进内存

train_dataset,test_dataset = amazon("%s/dataset.csv"%Config.envir_path).Split() 
mscnn = CNNnet()


topic_tokenizer = utils.CheckModel(
    'mscnn',
    lambda nl:mscnn.Train(train_dataset,test_dataset),
    continue_train=True,
    retrain=False
)
