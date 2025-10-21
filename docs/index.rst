Welcome to Text4GCN's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   contributing

Text4GCN is a Python framework for text classification using Graph Neural Networks (GNNs). 
It provides tools for building text-based graphs and applying Graph Convolutional Networks (GCNs) 
for document classification tasks.

Key Features
============

* **Multiple Dataset Support**: Built-in support for popular text classification datasets
* **Flexible Graph Building**: Various methods for constructing text graphs
* **GNN Models**: Pre-implemented Graph Neural Network architectures
* **Easy-to-use API**: Simple and intuitive interface for researchers and practitioners

Quick Start
===========

.. code-block:: python

   import text4gcn
   
   # Load a dataset
   data = text4gcn.datasets.R8(path="./data")
   
   # Build a text graph
   builder = text4gcn.builder.Frequency()
   adjacency = builder.tfidf(dataset_name="R8", dataset_path="./data")
   
   # Train a GCN model
   model = text4gcn.models.GNN(
       dataset="R8",
       path="./data",
       layer=text4gcn.models.Layer.GCN
   )
   model.fit()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
