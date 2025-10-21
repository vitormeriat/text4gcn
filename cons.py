from text4gcn.preprocess import TextPipeline
from text4gcn.models import Layer as layer
from text4gcn.datasets import data
import text4gcn.builder as builder
from text4gcn.models import GNN

# from text4gcn.models import Builder as bd


PIPE = False
TRAIN = False
TFIDF = False
WORD2VEC = False
COSIM = False
BUILDER = False
LWIC = False
BERT = True
LLM = False


builder.list_avaiable_ajacency_methods()

builder.list_avaiable_builders()

data.list()


if PIPE:

    data.R8(path="examples")
    # data.R52(path=path)
    # data.AG_NEWS(path=path)

    pipe = TextPipeline(
        dataset_name="R8",
        dataset_path="examples",
        rare_count=5,
        language="english")
    pipe.execute()


"""
To create the graph and its adjacency matrix, we use the builder module to import a specific 
model. Each model has a parameterization due to the text representation used in the RWF 
filter. In the example below we create two adjacency matrices. The first uses the 
EmbeddingAdjacency model and the second the DependencyParsingAdjacency model. For the 
EmbeddingAdjacency model based on Word2Vec, we inform the parameters for the training.
"""


frequency = builder.Frequency(dataset_name="R8", dataset_path="examples")
syntactic = builder.Syntactic(dataset_name="R8", dataset_path="examples")
semantic = builder.Semantic(dataset_name="R8", dataset_path="examples")


if LWIC:
    adj = syntactic.linguistic_inquiry(
        liwc_path="/home/vitormeriat/bin/LIWC2007_English100131.dic")

if TFIDF:
    adj = frequency.tfidf(window_size=20,
                          output_path=None)

if WORD2VEC:
    adj_w2v = semantic.word2vec(training_algorithm=1,
                                embedding_dimension=300,
                                num_epochs=20,
                                threshold=0.96,
                                language="english",
                                output_path=None)

if COSIM:
    adj_csim = semantic.cosine_similarity(threshold=0.9,
                                          output_path=None)

if BERT:
    adj = semantic.bert(language="english",
                        threshold=0.95)

if LLM:
    adj = semantic.llm()

if BUILDER:
    adj = syntactic.constituency_parsing()

    # export CLASSPATH=$CLASSPATH:/path/to/stanford-corenlp-4.5.6/*:
    adj = syntactic.dependency_parsing(core_nlp_path="/home/vitormeriat/bin/stanford-corenlp-4.5.6",
                                       language="english")

if TRAIN:
    gnn = GNN(
        dataset="R8",             # Dataset to train
        path="examples",          # Dataset path
        log_dir="examples/log",   # Log path
        layer=layer.GCN,          # Layer Type
        epoches=200,              # Number of traing epoches
        dropout=0.5,              # Dropout rate
        val_ratio=0.1,            # Train data used to validation
        early_stopping=10,        # Stop early technique
        lr=00.2,                  # Initial learing rate
        nhid=200,                 # Dimensions of hidden layers
        builder=builder.Word2Vec
    )

    gnn.fit()
