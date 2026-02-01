# 📚 Text4GCN - Sumário da Arquitetura e Módulos

## 🎯 Visão Geral

**Text4GCN** é um framework Python para classificação de texto usando Graph Neural Networks (GNNs). O framework combina técnicas de Processamento de Linguagem Natural (NLP) com Deep Learning em Grafos para criar representações gráficas de documentos e aplicar redes neurais convolucionais em grafos para tarefas de classificação.

### Características Principais

- 🚀 **API Fácil de Usar**: Interface simples e intuitiva para pesquisadores e profissionais
- 📊 **Suporte a Múltiplos Datasets**: Suporte integrado para datasets populares de classificação de texto (R8, R52, 20NG, etc.)
- 🔧 **Construção Flexível de Grafos**: Vários métodos para construir grafos de texto (semântico, sintático, baseado em frequência)
- 🧠 **Modelos GNN**: Arquiteturas de Graph Neural Network pré-implementadas (GCN, GAT)
- ⚡ **Otimizado para Memória**: Construção eficiente de grafos para datasets de texto em larga escala
- 🔬 **Pronto para Pesquisa**: Projetado para uso tanto em pesquisa quanto em produção

---

## 🏗️ Arquitetura do Framework

### Estrutura Modular

O Text4GCN segue uma arquitetura modular bem estruturada com 6 módulos principais:

```
text4gcn/
├── datasets/          # Gerenciamento de datasets
├── preprocess/        # Pipeline de pré-processamento
├── builder/           # Construção de matrizes de adjacência
├── models/            # Modelos de GNN
├── modules/           # Módulos utilitários
└── utils/             # Utilitários do sistema
```

### Camadas da Arquitetura

1. **Data Layer**: Carregamento e gerenciamento de datasets
2. **Preprocessing Layer**: Limpeza e preparação de dados de texto
3. **Builder Layer**: Construção de grafos usando diferentes estratégias
4. **Model Layer**: Implementação e treinamento de modelos GNN
5. **Utils Layer**: Gerenciamento de dispositivos, logging e operações de arquivo

---

## 📦 Módulos Detalhados

### 1. 📊 Módulo `datasets`

**Propósito**: Gerenciamento e carregamento de datasets de classificação de texto.

#### Classes Principais

- **`Datasets`**: Classe base para download e gerenciamento de datasets
- **`R8`**: Dataset Reuters com 8 classes
- **`R52`**: Dataset Reuters com 52 classes  
- **`AG_NEWS`**: Dataset de notícias AG

#### Métodos Principais

```python
# Listar datasets disponíveis
datasets.list()  # ['R8', 'R52', 'AG_NEWS']

# Download de dataset específico
datasets.R8(path="./data")
datasets.R52(path="./data")
datasets.AG_NEWS(path="./data")
```

#### Características dos Datasets

| Dataset | Docs | Training | Test | Words | Nodes | Classes | Avg Length |
|---------|------|----------|------|-------|-------|---------|------------|
| 20NG    | 18,846 | 11,314 | 7,532 | 42,757 | 61,603 | 20 | 221.26 |
| R8      | 7,674  | 5,485  | 2,189 | 7,688  | 15,362 | 8  | 65.72  |
| R52     | 9,100  | 6,532  | 2,568 | 8,892  | 17,992 | 52 | 69.82  |

### 2. 🔄 Módulo `preprocess`

**Propósito**: Pipeline de pré-processamento de dados de texto.

#### Classes Principais

- **`TextPipeline`**: Pipeline principal de pré-processamento
- **`PreProcessingConfig`**: Configurações de pré-processamento

#### Métodos do TextPipeline

```python
class TextPipeline:
    def __init__(self, dataset_name: str, dataset_path: str, 
                 rare_count: int = 5, language: str = "english")
    
    def execute(self) -> None:
        """Executa o pipeline completo de pré-processamento"""
    
    def clean_data(self) -> None:
        """Remove palavras raras e stop words"""
    
    def shuffle_data(self) -> None:
        """Embaralha os dados do corpus"""
```

#### Processo de Pré-processamento

1. **Limpeza de Dados**: Remove caracteres especiais e normaliza texto
2. **Remoção de Stop Words**: Remove palavras irrelevantes
3. **Filtragem de Palavras Raras**: Remove palavras com baixa frequência
4. **Embaralhamento**: Reorganiza dados para treinamento/teste

### 3. 🔧 Módulo `builder`

**Propósito**: Construção de matrizes de adjacência usando diferentes estratégias.

#### Estratégias de Construção

##### 🧠 **Semantic Builders**
- **`BERTAdjacency`**: Usa embeddings BERT para similaridade semântica
- **`Word2VecAdjacency`**: Usa Word2Vec para relações semânticas
- **`CosineSimilarityAdjacency`**: Similaridade por cosseno entre vetores

##### 🌳 **Syntactic Builders**
- **`DependencyParsingAdjacency`**: Usa Stanford CoreNLP para dependências sintáticas
- **`LiwcAdjacency`**: Usa LIWC (Linguistic Inquiry and Word Count)
- **`ConstituencyParsingAdjacency`**: Análise de constituintes sintáticos

##### 📈 **Frequency Builders**
- **`FrequencyAdjacency`**: Usa TF-IDF para pesos estatísticos

#### Classes Principais

```python
class Frequency:
    def tfidf(self) -> None:
        """Constrói matriz usando TF-IDF"""

class Semantic:
    def bert(self) -> None:
        """Constrói matriz usando BERT"""
    def word2vec(self) -> None:
        """Constrói matriz usando Word2Vec"""
    def cosine_similarity(self) -> None:
        """Constrói matriz usando similaridade por cosseno"""

class Syntactic:
    def dependency_parsing(self) -> None:
        """Constrói matriz usando parsing de dependências"""
    def constituency_parsing(self) -> None:
        """Constrói matriz usando parsing de constituintes"""
```

### 4. 🧠 Módulo `models`

**Propósito**: Implementação e treinamento de modelos Graph Neural Network.

#### Classes Principais

- **`GNN`**: Classe principal para modelos GNN
- **`GCN`**: Implementação de Graph Convolutional Network
- **`GraphConvolution`**: Camada de convolução em grafos
- **`TextGCNTrainer`**: Treinador para modelos TextGCN

#### Arquitetura do Modelo

```python
class GCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 support: list, num_classes: int):
        self.layer1 = GraphConvolution(input_dim, output_dim, support, 
                                      act_func=nn.ReLU(), featureless=True)
        self.layer2 = GraphConvolution(output_dim, num_classes, support)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
```

#### Configuração do Modelo

```python
model = text4gcn.models.GNN(
    dataset="R8",
    path="./data",
    log_dir="./logs",
    builder=text4gcn.builder.Builder.Tfidf,
    layer=text4gcn.models.Layer.GCN,
    epochs=200,
    dropout=0.5,
    device=None  # Auto-detect
)
```

### 5. 🛠️ Módulo `modules`

**Propósito**: Módulos utilitários para operações auxiliares.

#### Submódulos

##### **`file_ops.py`**
- **`FileOps`**: Operações de arquivo e diretório
- Métodos: `create_dir()`, `check_paths()`, `write_iterable_to_file()`

##### **`logger.py`**
- **`PrintLog`**: Sistema de logging
- **`Process`**: Decorador para logging de processos
- Métodos: `info()`, `warning()`, `error()`, `log_history()`

##### **`clean_data.py`**
- **`clean_str()`**: Limpeza de strings
- **`extract_word_counts()`**: Extração de contagens de palavras
- **`remove_stop_words()`**: Remoção de stop words
- **`remove_rare_words()`**: Remoção de palavras raras

##### **`node_features.py`**
- **`NodeFeatures`**: Extração de características de nós
- Métodos: `extract_doc_labels()`, `compute_x()`, `compute_y()`

##### **`word_processor.py`**
- **`extract_vocabulary()`**: Extração de vocabulário
- **`word_to_vectors()`**: Conversão de palavras para vetores
- **`extract_tf_idf_word_vectors()`**: Extração de vetores TF-IDF

##### **`adjacency.py`**
- **`extract_word_to_doc_ids()`**: Mapeamento palavra-documento
- **`flatten_nested_iterables()`**: Flattening de estruturas aninhadas

##### **`textvec/`**
- **`word_features.py`**: Treinamento e inferência Word2Vec
- **`liwc.py`**: Análise LIWC

### 6. 🔧 Módulo `utils`

**Propósito**: Utilitários do sistema e gerenciamento de dispositivos.

#### Classes Principais

- **`DeviceManager`**: Gerenciamento automático de dispositivos (CPU/GPU)
- **`check_pytorch_installation()`**: Verificação de instalação PyTorch
- **`recommend_installation()`**: Recomendações de instalação

#### Funcionalidades

```python
# Verificação de instalação
info = text4gcn.utils.check_pytorch_installation()
print(f"PyTorch available: {info['torch_available']}")
print(f"CUDA available: {info['cuda_available']}")

# Gerenciamento de dispositivo
device_manager = text4gcn.utils.get_device_manager()
print(f"Using device: {device_manager.device}")
```

---

## 🔄 Fluxo de Dados

### 1. **Carregamento de Dados**
```python
# Carregar dataset
data = text4gcn.datasets.R8(path="./data")
```

### 2. **Pré-processamento**
```python
# Pipeline de pré-processamento
pipe = text4gcn.preprocess.TextPipeline(
    dataset_name="R8",
    dataset_path="./data",
    rare_count=5,
    language="english"
)
pipe.execute()
```

### 3. **Construção de Grafo**
```python
# Construir matriz de adjacência
builder = text4gcn.builder.Frequency()
adjacency = builder.tfidf(dataset_name="R8", dataset_path="./data")
```

### 4. **Treinamento do Modelo**
```python
# Treinar modelo GNN
model = text4gcn.models.GNN(
    dataset="R8",
    path="./data",
    builder=text4gcn.builder.Builder.Tfidf,
    layer=text4gcn.models.Layer.GCN
)
model.fit()
```

---

## 🎯 API Principal

### Importação Básica
```python
import text4gcn
```

### Módulos Disponíveis
```python
# Acesso aos módulos
text4gcn.datasets    # Gerenciamento de datasets
text4gcn.preprocess  # Pré-processamento
text4gcn.builder     # Construção de grafos
text4gcn.models      # Modelos GNN
text4gcn.modules     # Módulos utilitários
text4gcn.utils       # Utilitários do sistema
```

### Exemplo Completo
```python
import text4gcn

# 1. Verificar instalação
info = text4gcn.utils.check_pytorch_installation()

# 2. Carregar dataset
data = text4gcn.datasets.R8(path="./data")

# 3. Pré-processar
pipe = text4gcn.preprocess.TextPipeline("R8", "./data")
pipe.execute()

# 4. Construir grafo
builder = text4gcn.builder.Frequency()
builder.tfidf(dataset_name="R8", dataset_path="./data")

# 5. Treinar modelo
model = text4gcn.models.GNN(
    dataset="R8",
    path="./data",
    log_dir="./logs",
    builder=text4gcn.builder.Builder.Tfidf,
    layer=text4gcn.models.Layer.GCN,
    epochs=200,
    dropout=0.5
)
model.fit()
```

---

## 🔧 Configuração e Dependências

### Dependências Principais

```toml
# Core dependencies
transformers = "^4.41.1"
scikit-learn = "^1.3.2"
numpy = "^1.24.0"
pandas = "^2.0.0"

# NLP dependencies
nltk = "^3.8.1"
gensim = "^4.3.2"
stanfordcorenlp = "^3.9.1.1"

# Visualization and utilities
matplotlib = "^3.8.2"
tabulate = "^0.9.0"
rich = "^13.0.0"
```

### Suporte a Versões Python

- **Python 3.10** ✅
- **Python 3.11** ✅ (Recomendado)
- **Python 3.12** ✅
- **Python 3.13** ✅

### Instalação

```bash
# CPU version
pip install text4gcn[cpu]

# GPU version
pip install text4gcn[gpu]

# Auto-detect
pip install text4gcn
```

---

## 📊 Métricas e Performance

### Estrutura de Dados

- **Matrizes Esparsas**: Uso eficiente de memória com `scipy.sparse`
- **Batch Processing**: Processamento em lotes para datasets grandes
- **Device Management**: Suporte automático a CPU/GPU

### Otimizações

- **Memory Optimization**: Construção eficiente de grafos
- **Sparse Matrices**: Representação compacta de adjacências
- **Vectorized Operations**: Operações vetorizadas com NumPy/PyTorch

---

## 🚀 Extensibilidade

### Adicionando Novos Builders

```python
class CustomAdjacency:
    def __init__(self, dataset_name: str, dataset_path: str):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
    
    def build(self):
        # Implementar lógica de construção
        pass
```

### Adicionando Novos Modelos

```python
class CustomGNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_classes: int):
        super().__init__()
        # Implementar arquitetura customizada
    
    def forward(self, x):
        # Implementar forward pass
        return output
```

---

## 📈 Casos de Uso

### 1. **Pesquisa Acadêmica**
- Experimentos com diferentes estratégias de construção de grafos
- Comparação de arquiteturas GNN
- Análise de datasets de classificação de texto

### 2. **Aplicações Industriais**
- Classificação de documentos
- Análise de sentimento
- Categorização de conteúdo

### 3. **Desenvolvimento de Produto**
- Sistemas de recomendação baseados em texto
- Análise de feedback de usuários
- Classificação automática de conteúdo

---

## 🔍 Troubleshooting

### Problemas Comuns

1. **Erro de Memória**: Use datasets menores ou processamento em lotes
2. **Dependências**: Verifique instalação do PyTorch e CUDA
3. **Paths**: Certifique-se de que os caminhos dos datasets estão corretos

### Debug

```python
# Verificar instalação
text4gcn.utils.check_pytorch_installation()

# Verificar dispositivo
device_manager = text4gcn.utils.get_device_manager()
print(device_manager.get_device_info())
```

---

## 📚 Recursos Adicionais

### Documentação
- **README.md**: Guia de início rápido
- **docs/**: Documentação completa
- **examples/**: Exemplos de uso

### Comunidade
- **GitHub**: https://github.com/vitormeriat/text4gcn
- **Issues**: Reportar bugs e solicitar features
- **Discussions**: Discussões da comunidade

### Contribuição
- Fork do repositório
- Criação de branch para feature
- Pull request com descrição detalhada

---

**Última atualização**: Setembro 2024  
**Versão**: 2.0.0  
**Status**: ✅ Ativo - Suporte completo para Python 3.10-3.13
