from text4gcn.models import Builder as bd
from text4gcn.models import Layer as layer
from text4gcn.models import GNN
from text4gcn.preprocess import TextPipeline
from text4gcn.builder import *
from text4gcn.datasets import data


print(f"\n{'='*60} START OF TEST\n")

path = "examples"

print(data.list())

data.R8(path=path)
# data.R52(path=path)
# data.AG_NEWS(path=path)

print("OK")

# #print(help(layer))

# ======================= TextPipeline
pipe = TextPipeline(
    dataset_name="R8",
    rare_count=5,
    dataset_path=path,
    language="english")
pipe.execute()
# =======================

# # ======================= FrequencyAdjacency
# freq = FrequencyAdjacency(
#     dataset_name="R8",
#     dataset_path=path
# )
# freq.build()

# # ======================= CosineSimilarityAdjacency
# freq = CosineSimilarityAdjacency(
#     dataset_name="R8",
#     dataset_path=path
# )
# freq.build()

# ======================= EmbeddingAdjacency
freq = EmbeddingAdjacency(
    dataset_name="R8",
    dataset_path=path,
    num_epochs=20,
    embedding_dimension=300,
    training_regime=1
)
freq.build()

# # ======================= DependencyParsingAdjacency
# freq = DependencyParsingAdjacency(
#     dataset_name="R8",
#     dataset_path=path,
#     core_nlp_path="C:/bin/CoreNLP/stanford-corenlp-full-2018-10-05"
# )
# # freq.build()

# # ======================= ConstituencyParsingAdjacency
# # freq = ConstituencyParsingAdjacency()


# # ======================= LiwcAdjacency
# freq = LiwcAdjacency(
#     dataset_name="R8",
#     dataset_path=path,
#     liwc_path="ztst/LIWC2007_English100131.dic"
# )
# freq.build()


gnn = GNN(
    dataset="R8",           # Dataset to train
    path=path,              # Dataset path
    log_dir="examples/log", # Log path
    layer=layer.GCN,        # Layer Type
    epoches=200,            # Number of traing epoches
    dropout=0.5,            # Dropout rate
    val_ratio=0.1,          # Train data used to validation
    early_stopping=10,      # Stop early technique
    lr=00.2,                # Initial learing rate
    nhid=200,               # Dimensions of hidden layers
    builder=bd.Embedding    # Type of Filtered Text Graph
)
gnn.fit()


print(f"\n{'='*60} END OF TEST\n")
