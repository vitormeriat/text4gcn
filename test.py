from text4gcn.preprocess import TextPipeline
from text4gcn.builder import *
from text4gcn.models import GNN
from text4gcn.models import Layer as layer
from text4gcn.datasets import data


print(f"\n{'='*60} START OF TEST\n")

path = "ztst"

print(data.list())

data.R8(path=path)
#data.R52(path=path)
#data.AG_NEWS(path=path)

print("OK")

# # ======================= TextPipeline
pipe = TextPipeline(
    dataset_name="R8",
    rare_count=5,
    dataset_path=path,
    language="english")

pipe.execute()
# # =======================

# # ======================= FrequencyAdjacency
freq = FrequencyAdjacency(
    dataset_name="R8",
    dataset_path=path
)
freq.build()

# # ======================= CosineSimilarityAdjacency
freq = CosineSimilarityAdjacency(
    dataset_name="R8",
    dataset_path=path
)
# freq.build()

# # ======================= EmbeddingAdjacency
freq = EmbeddingAdjacency(
    dataset_name="test",
    dataset_path=path,
    num_epochs=20,
    embedding_dimension=300,
    training_regime=1
)
# freq.build()

# # ======================= DependencyParsingAdjacency
freq = DependencyParsingAdjacency(
    dataset_name="test",
    dataset_path=path,
    core_nlp_path="C:/bin/CoreNLP/stanford-corenlp-full-2018-10-05"
)
# freq.build()

# # ======================= ConstituencyParsingAdjacency
# # freq = ConstituencyParsingAdjacency()


# # ======================= LiwcAdjacency
freq = LiwcAdjacency(
    dataset_name="test",
    dataset_path=path,
    liwc_path="C:/bin/LIWC/LIWC2007_English100131.dic"
)
# freq.build()


gnn = GNN(
    dataset="R8",
    path=path,
    log_dir="log",
    layer=layer.GCN,
    epoches=200,
    dropout=0.5,
    val_ratio=0.1,
    early_stopping=10,
    lr=00.2,
    nhid=200
)

#gnn.fit()


print(f"\n{'='*60} END OF TEST\n")


# class Person:
#     """
#     A class to represent a person.

#     ...

#     Attributes
#     ----------
#     name : str
#         first name of the person
#     surname : str
#         family name of the person
#     age : int
#         age of the person

#     Methods
#     -------
#     info(additional=""):
#         Prints the person's name and age.
#     """

#     def __init__(self, name, surname, age):
#         """
#         Constructs all the necessary attributes for the person object.

#         Parameters
#         ----------
#             name : str
#                 first name of the person
#             surname : str
#                 family name of the person
#             age : int
#                 age of the person
#         """

#         self.name = name
#         self.surname = surname
#         self.age = age

#     def info(self, additional=""):
#         """
#         Prints the person's name and age.

#         If the argument 'additional' is passed, then it is appended after the main info.

#         Parameters
#         ----------
#         additional : str, optional
#             More info to be displayed (default is None)

#         Returns
#         -------
#         None
#         """

#         print(f'My name is {self.name} {self.surname}. I am {self.age} years old.' + additional)
