# Diagramas de Arquitetura do Text4GCN

## 1. Arquitetura Geral do Text4GCN

```mermaid
graph TB
    subgraph "Text4GCN Framework"
        subgraph "Data Layer"
            DS[Datasets Module]
            R8[R8 Dataset]
            R52[R52 Dataset] 
            AG[AG_NEWS Dataset]
        end
        
        subgraph "Preprocessing Layer"
            TP[TextPipeline]
            CD[Clean Data]
            SW[Stop Words]
            RW[Rare Words]
            SF[Shuffle Data]
        end
        
        subgraph "Builder Layer"
            subgraph "Semantic Builders"
                BERT[BERT Adjacency]
                W2V[Word2Vec Adjacency]
                CS[Cosine Similarity]
            end
            
            subgraph "Syntactic Builders"
                DP[Dependency Parsing]
                CP[Constituency Parsing]
                LIWC[LIWC Adjacency]
            end
            
            subgraph "Frequency Builders"
                TFIDF[TF-IDF Adjacency]
            end
        end
        
        subgraph "Model Layer"
            GNN[GNN Model]
            GCN[GCN Layer]
            GAT[GAT Layer]
            TE[TextGCNTrainer]
        end
        
        subgraph "Utils Layer"
            DM[Device Manager]
            LOG[Logger]
            FO[File Operations]
        end
    end
    
    DS --> TP
    TP --> CD
    CD --> SW
    SW --> RW
    RW --> SF
    
    SF --> BERT
    SF --> W2V
    SF --> CS
    SF --> DP
    SF --> CP
    SF --> LIWC
    SF --> TFIDF
    
    BERT --> GNN
    W2V --> GNN
    CS --> GNN
    DP --> GNN
    CP --> GNN
    LIWC --> GNN
    TFIDF --> GNN
    
    GNN --> GCN
    GNN --> GAT
    GCN --> TE
    GAT --> TE
    
    DM --> GNN
    LOG --> TP
    LOG --> BERT
    LOG --> W2V
    LOG --> CS
    LOG --> DP
    LOG --> CP
    LOG --> LIWC
    LOG --> TFIDF
    LOG --> TE
    FO --> TP
    FO --> BERT
    FO --> W2V
    FO --> CS
    FO --> DP
    FO --> CP
    FO --> LIWC
    FO --> TFIDF
```

## 2. Integração entre Módulos

```mermaid
graph LR
    subgraph "text4gcn Package"
        subgraph "datasets/"
            D1[data.py]
            D2[__init__.py]
        end
        
        subgraph "preprocess/"
            P1[pipeline.py]
            P2[config.py]
            P3[__init__.py]
        end
        
        subgraph "builder/"
            B1[pipeline.py]
            B2[frequency_adj.py]
            B3[word2vec_adj.py]
            B4[bert_adj.py]
            B5[cosine_similarity_adj.py]
            B6[dependency_parsing_adj.py]
            B7[linguistic_inquiry_adj.py]
            B8[__init__.py]
        end
        
        subgraph "models/"
            M1[gnn.py]
            M2[model.py]
            M3[train_eval.py]
            M4[utils.py]
            M5[layers.py]
            M6[builder.py]
            M7[__init__.py]
        end
        
        subgraph "modules/"
            MO1[file_ops.py]
            MO2[logger.py]
            MO3[clean_data.py]
            MO4[node_features.py]
            MO5[word_processor.py]
            MO6[adjacency.py]
            MO7[utils.py]
            MO8[__init__.py]
        end
        
        subgraph "utils/"
            U1[device.py]
            U2[__init__.py]
        end
    end
    
    D1 --> P1
    P1 --> MO1
    P1 --> MO2
    P1 --> MO3
    P1 --> MO4
    
    P1 --> B1
    B1 --> B2
    B1 --> B3
    B1 --> B4
    B1 --> B5
    B1 --> B6
    B1 --> B7
    
    B2 --> MO1
    B3 --> MO1
    B4 --> MO1
    B5 --> MO1
    B6 --> MO1
    B7 --> MO1
    
    B2 --> MO2
    B3 --> MO2
    B4 --> MO2
    B5 --> MO2
    B6 --> MO2
    B7 --> MO2
    
    B1 --> M1
    M1 --> M2
    M1 --> M3
    M1 --> M4
    M1 --> M5
    M1 --> M6
    
    M1 --> U1
    M3 --> MO1
    M3 --> MO2
```

## 3. Fluxo de Dados

```mermaid
flowchart TD
    Start([Início]) --> LoadData[Carregar Dataset]
    LoadData --> TextPipeline[TextPipeline.execute()]
    
    TextPipeline --> CleanData[clean_data()]
    CleanData --> RemoveStopWords[remove_stop_words()]
    RemoveStopWords --> RemoveRareWords[remove_rare_words()]
    RemoveRareWords --> ShuffleData[shuffle_data()]
    
    ShuffleData --> ChooseBuilder{Escolher Builder}
    
    ChooseBuilder -->|Semantic| SemanticBuilder[Semantic Builder]
    ChooseBuilder -->|Syntactic| SyntacticBuilder[Syntactic Builder]
    ChooseBuilder -->|Frequency| FrequencyBuilder[Frequency Builder]
    
    SemanticBuilder --> BERTMethod[BERT Method]
    SemanticBuilder --> Word2VecMethod[Word2Vec Method]
    SemanticBuilder --> CosineMethod[Cosine Similarity Method]
    
    SyntacticBuilder --> DependencyMethod[Dependency Parsing Method]
    SyntacticBuilder --> ConstituencyMethod[Constituency Parsing Method]
    SyntacticBuilder --> LIWCMethod[LIWC Method]
    
    FrequencyBuilder --> TFIDFMethod[TF-IDF Method]
    
    BERTMethod --> BuildAdjacency[build_adjacency_matrix()]
    Word2VecMethod --> BuildAdjacency
    CosineMethod --> BuildAdjacency
    DependencyMethod --> BuildAdjacency
    ConstituencyMethod --> BuildAdjacency
    LIWCMethod --> BuildAdjacency
    TFIDFMethod --> BuildAdjacency
    
    BuildAdjacency --> CreateGNN[GNN Model Creation]
    CreateGNN --> LoadData[LoadData.load_corpus()]
    LoadData --> PrepareData[prepare_data()]
    PrepareData --> ConvertTensor[convert_tensor()]
    ConvertTensor --> TrainModel[train_model()]
    
    TrainModel --> EvaluateModel[evaluate_model()]
    EvaluateModel --> End([Fim])
```

## 4. Métodos de Construção de Adjacência

```mermaid
graph TB
    subgraph "Adjacency Matrix Builders"
        subgraph "Semantic Approaches"
            BERT[BERT Adjacency<br/>- Uses BERT embeddings<br/>- Semantic similarity<br/>- Threshold-based filtering]
            
            W2V[Word2Vec Adjacency<br/>- Word embeddings<br/>- Semantic relationships<br/>- Vector similarity]
            
            CS[Cosine Similarity<br/>- Vector similarity<br/>- Cosine distance<br/>- Threshold filtering]
        end
        
        subgraph "Syntactic Approaches"
            DP[Dependency Parsing<br/>- Stanford CoreNLP<br/>- Syntactic dependencies<br/>- Parse tree relationships]
            
            CP[Constituency Parsing<br/>- Parse tree structure<br/>- Constituent relationships<br/>- Tree-based connections]
            
            LIWC[LIWC Adjacency<br/>- Linguistic Inquiry<br/>- Word Count analysis<br/>- Psychological categories]
        end
        
        subgraph "Frequency Approaches"
            TFIDF[TF-IDF Adjacency<br/>- Term frequency<br/>- Inverse document frequency<br/>- Statistical weighting]
        end
    end
    
    subgraph "Common Process"
        LoadCorpus[Load Corpus]
        ExtractVocab[Extract Vocabulary]
        BuildRelations[Build Word Relations]
        ComputeWeights[Compute Edge Weights]
        CreateMatrix[Create Sparse Matrix]
        SaveMatrix[Save Adjacency Matrix]
    end
    
    BERT --> LoadCorpus
    W2V --> LoadCorpus
    CS --> LoadCorpus
    DP --> LoadCorpus
    CP --> LoadCorpus
    LIWC --> LoadCorpus
    TFIDF --> LoadCorpus
    
    LoadCorpus --> ExtractVocab
    ExtractVocab --> BuildRelations
    BuildRelations --> ComputeWeights
    ComputeWeights --> CreateMatrix
    CreateMatrix --> SaveMatrix
```

## 5. Estrutura de Classes e Herança

```mermaid
classDiagram
    class Adjacency {
        +dataset_name: str
        +dataset_path: str
        +__init__(dataset_name, dataset_path)
    }
    
    class Frequency {
        +__init__(dataset_name, dataset_path)
        +tfidf()
    }
    
    class Semantic {
        +__init__(dataset_name, dataset_path)
        +bert()
        +word2vec()
        +cosine_similarity()
    }
    
    class Syntactic {
        +__init__(dataset_name, dataset_path)
        +dependency_parsing()
        +constituency_parsing()
    }
    
    class FrequencyAdjacency {
        +dataset_name: str
        +dataset_path: str
        +logger: PrintLog
        +flop: FileOps
        +build()
    }
    
    class BERTAdjacency {
        +dataset_name: str
        +dataset_path: str
        +threshold: float
        +language: str
        +build()
    }
    
    class Word2VecAdjacency {
        +dataset_name: str
        +dataset_path: str
        +build()
    }
    
    class DependencyParsingAdjacency {
        +dataset_name: str
        +dataset_path: str
        +core_nlp_path: str
        +language: str
        +build()
    }
    
    class LiwcAdjacency {
        +dataset_name: str
        +dataset_path: str
        +liwc_path: str
        +build()
    }
    
    class CosineSimilarityAdjacency {
        +dataset_name: str
        +dataset_path: str
        +threshold: float
        +build()
    }
    
    class GNN {
        +dataset: str
        +path: str
        +log_dir: str
        +builder: str
        +layer: Layer
        +epoches: int
        +dropout: float
        +device: str
        +config()
        +fit()
    }
    
    class TextGCNTrainer {
        +model: nn.Module
        +args: argparse.Namespace
        +predata: dict
        +fit()
        +prepare_data()
        +train_model()
    }
    
    class GCN {
        +input_dim: int
        +output_dim: int
        +support: list
        +num_classes: int
        +forward(x)
    }
    
    class GraphConvolution {
        +input_dim: int
        +output_dim: int
        +support: list
        +dropout_rate: float
        +forward(x)
    }
    
    Adjacency <|-- Frequency
    Adjacency <|-- Semantic
    Adjacency <|-- Syntactic
    
    Frequency --> FrequencyAdjacency
    Semantic --> BERTAdjacency
    Semantic --> Word2VecAdjacency
    Semantic --> CosineSimilarityAdjacency
    Syntactic --> DependencyParsingAdjacency
    Syntactic --> LiwcAdjacency
    
    GNN --> TextGCNTrainer
    TextGCNTrainer --> GCN
    GCN --> GraphConvolution
```

