import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
from torch.utils.data import DataLoader

#python training.py model_name_path file_path

os.makedirs('output', exist_ok=True)
os.makedirs('data', exist_ok=True)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_name = sys.argv[1] if len(sys.argv) > 1 else 'distilroberta-base'
num_epochs = 3
sts_dataset_path = 'data/stsbenchmark.tsv.gz'
batch_size_pairs = 384
batch_size_triplets = 256
train_batch_size=16
max_seq_length = 128
use_amp = True                  #Set to False, if you use a CPU or your GPU does not support FP16 operations
evaluation_steps = 500
# warmup_steps = 500

#####



if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


# Save path of the model
model_save_path = 'output/training_paraphrases_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



## SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

datasets = []
for filepath in sys.argv[2:]:
    dataset = []
    with_guid = 'with-guid' in filepath     #Some datasets have a guid in the first column

    with gzip.open(filepath, 'rt', encoding='utf8') as fIn:
        for line in fIn:
            splits = line.strip().split("\t")
            if with_guid:
                guid = splits[0]
                texts = splits[1:]
            else:
                guid = None
                texts = splits

            dataset.append(InputExample(texts=texts, guid=guid))

    datasets.append(dataset)

# Convert the dataset to a DataLoader ready for training
logging.info("Read train dataset")

train_dataloader = DataLoader(dataset, shuffle=True, batch_size=train_batch_size)
# Our training loss
train_loss = losses.MultipleNegativesRankingLoss(model)

#Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
test_samples=[]
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
        inp_example=InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
        
        if row['split'] == 'dev':
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            test_samples.append(inp_example)
            

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=use_amp,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=1000,
          checkpoint_save_total_limit=3
          )

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)