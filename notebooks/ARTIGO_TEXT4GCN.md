# Text4GCN: Um Framework para Classificação de Texto Baseado em Redes Neurais Convolucionais em Grafos

## Resumo

Este artigo apresenta o **Text4GCN**, um framework Python de código aberto para classificação de texto utilizando Redes Neurais Convolucionais em Grafos (GCN). O framework combina técnicas de Processamento de Linguagem Natural (PLN) com algoritmos de Deep Learning em Grafos para construir representações gráficas de documentos e aplicar redes neurais convolucionais em grafos para tarefas de classificação. A principal contribuição deste trabalho é fornecer um framework flexível capaz de realizar filtragem sintática e semântica que torna os grafos de texto menores e mais representativos, oferecendo uma ferramenta alternativa e poderosa para o estudo de Redes Neurais Convolucionais em Grafos em tarefas de classificação de texto.

**Palavras-chave:** Classificação de Texto, Redes Neurais Convolucionais em Grafos, Processamento de Linguagem Natural, Deep Learning, Grafos Heterogêneos

## 1. Introdução

A classificação de texto é uma das tarefas fundamentais no Processamento de Linguagem Natural (PLN), com aplicações que vão desde filtragem de spam e análise de sentimentos até categorização de documentos e sistemas de recomendação. Métodos tradicionais de classificação de texto frequentemente tratam documentos como vetores independentes em um espaço de alta dimensionalidade, ignorando as relações semânticas e sintáticas entre palavras e documentos.

Recentemente, Redes Neurais Convolucionais em Grafos (GCNs) têm demonstrado sucesso significativo em tarefas de classificação de texto, superando métodos tradicionais ao capturar relações estruturais entre palavras e documentos [1, 2]. No entanto, a construção eficiente de grafos de texto que capturem essas relações de forma representativa permanece um desafio.

O **Text4GCN** aborda essa limitação fornecendo um framework flexível que permite a construção de grafos de texto otimizados em memória utilizando diferentes representações textuais. O framework oferece múltiplos métodos para construir matrizes de adjacência baseados em:

- **Abordagens Semânticas**: BERT, Word2Vec, Similaridade de Cosseno
- **Abordagens Sintáticas**: Dependency Parsing, Constituency Parsing, LIWC
- **Abordagens Baseadas em Frequência**: TF-IDF

Este artigo descreve a arquitetura, metodologia e implementação do Text4GCN, demonstrando como diferentes estratégias de construção de grafos podem ser aplicadas para melhorar o desempenho em tarefas de classificação de texto.

## 2. Fundamentação Teórica

### 2.1 Redes Neurais Convolucionais em Grafos

As Redes Neurais Convolucionais em Grafos (GCNs) foram introduzidas por Kipf e Welling [1] como uma extensão das redes neurais convolucionais para dados estruturados em grafos. Dado um grafo $G = (V, E)$ com $N$ nós, onde $V$ é o conjunto de vértices e $E$ é o conjunto de arestas, uma camada de convolução em grafo pode ser definida como:

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right)$$

onde:
- $\tilde{A} = A + I_N$ é a matriz de adjacência com auto-conexões adicionadas
- $I_N$ é a matriz identidade
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ é a matriz de grau
- $H^{(l)}$ são as características dos nós na camada $l$
- $W^{(l)}$ são os pesos treináveis da camada $l$
- $\sigma(\cdot)$ é uma função de ativação (tipicamente ReLU)

### 2.2 Grafos Heterogêneos para Texto

No contexto de classificação de texto, construímos um grafo heterogêneo $G = (V, E)$ onde:
- $V = V_d \cup V_w$ é o conjunto de nós, composto por nós de documentos $V_d$ e nós de palavras $V_w$
- $E$ é o conjunto de arestas que conectam documentos a palavras e palavras entre si

A matriz de adjacência $A$ é definida como:

$$A = \begin{bmatrix}
A_{dd} & A_{dw} \\
A_{wd} & A_{ww}
\end{bmatrix}$$

onde:
- $A_{dd}$ representa conexões documento-documento (geralmente vazia)
- $A_{dw}$ e $A_{wd}$ representam conexões documento-palavra (simétricas)
- $A_{ww}$ representa conexões palavra-palavra

### 2.3 Métodos de Construção de Adjacência

#### 2.3.1 TF-IDF (Term Frequency-Inverse Document Frequency)

O TF-IDF é uma medida estatística que avalia a importância de uma palavra em um documento relativa a uma coleção de documentos. O peso TF-IDF entre um documento $d$ e uma palavra $w$ é calculado como:

$$\text{TF-IDF}(d, w) = \text{TF}(d, w) \times \text{IDF}(w)$$

onde:

$$\text{TF}(d, w) = \frac{\text{count}(w, d)}{\text{total words in } d}$$

$$\text{IDF}(w) = \log\left(\frac{N}{\text{df}(w)}\right)$$

sendo $N$ o número total de documentos e $\text{df}(w)$ o número de documentos que contêm a palavra $w$.

A matriz de adjacência é construída com pesos $A_{ij} = \text{TF-IDF}(d_i, w_j)$ para conexões documento-palavra.

#### 2.3.2 Word2Vec

Word2Vec [3] gera embeddings densos de palavras que capturam relações semânticas. Para construir a matriz de adjacência, calculamos a similaridade entre embeddings de palavras:

$$A_{ww}(i, j) = \begin{cases}
\text{sim}(\mathbf{v}_i, \mathbf{v}_j) & \text{se } \text{sim}(\mathbf{v}_i, \mathbf{v}_j) > \theta \\
0 & \text{caso contrário}
\end{cases}$$

onde $\mathbf{v}_i$ e $\mathbf{v}_j$ são os embeddings das palavras $i$ e $j$, $\text{sim}(\cdot, \cdot)$ é uma medida de similaridade (tipicamente cosseno), e $\theta$ é um threshold.

#### 2.3.3 BERT

Similar ao Word2Vec, utilizamos embeddings BERT [4] para capturar relações semânticas mais ricas. A similaridade entre palavras é calculada usando embeddings contextuais do BERT:

$$A_{ww}(i, j) = \begin{cases}
\text{cosine}(\text{BERT}(w_i), \text{BERT}(w_j)) & \text{se } \text{cosine}(\cdot, \cdot) > \theta \\
0 & \text{caso contrário}
\end{cases}$$

#### 2.3.4 Dependency Parsing

Dependency Parsing captura relações sintáticas entre palavras baseadas em árvores de dependência. A matriz de adjacência é construída conectando palavras que têm relações de dependência direta:

$$A_{ww}(i, j) = \begin{cases}
1 & \text{se existe relação de dependência entre } w_i \text{ e } w_j \\
0 & \text{caso contrário}
\end{cases}$$

#### 2.3.5 Similaridade de Cosseno

A similaridade de cosseno entre vetores de palavras é calculada como:

$$\text{cosine}(\mathbf{v}_i, \mathbf{v}_j) = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{||\mathbf{v}_i|| \cdot ||\mathbf{v}_j||}$$

A matriz de adjacência é construída aplicando um threshold $\theta$:

$$A_{ww}(i, j) = \begin{cases}
\text{cosine}(\mathbf{v}_i, \mathbf{v}_j) & \text{se } \text{cosine}(\mathbf{v}_i, \mathbf{v}_j) > \theta \\
0 & \text{caso contrário}
\end{cases}$$

## 3. Arquitetura do Framework Text4GCN

O Text4GCN segue uma arquitetura modular composta por cinco camadas principais:

### 3.1 Camada de Dados (Data Layer)

Responsável pelo carregamento e gerenciamento de datasets. O framework suporta múltiplos datasets de benchmark, incluindo:
- **R8**: Reuters News Dataset com 8 classes
- **R52**: Reuters News Dataset com 52 classes  
- **20NG**: 20 Newsgroups Dataset
- **AG_NEWS**: AG News Dataset

### 3.2 Camada de Pré-processamento (Preprocessing Layer)

Realiza o processamento inicial dos textos através do módulo `TextPipeline`, que executa as seguintes etapas:

1. **Limpeza de Dados**: Remoção de caracteres especiais e normalização
2. **Remoção de Stop Words**: Eliminação de palavras comuns que não carregam informação semântica significativa
3. **Remoção de Palavras Raras**: Filtragem de palavras que aparecem com frequência muito baixa (threshold configurável)
4. **Embaralhamento**: Reorganização aleatória dos dados para treinamento

O pipeline gera os seguintes arquivos:
- Corpus tokenizado e limpo
- Vocabulário processado
- Índices de treinamento, validação e teste
- Vetores de palavras (quando aplicável)

### 3.3 Camada de Construção (Builder Layer)

Esta camada é responsável pela construção das matrizes de adjacência utilizando diferentes estratégias. O framework implementa os seguintes builders:

#### Builders Semânticos
- **BERTAdjacency**: Utiliza embeddings BERT para capturar relações semânticas
- **Word2VecAdjacency**: Aplica Word2Vec para gerar embeddings de palavras
- **CosineSimilarityAdjacency**: Calcula similaridade de cosseno entre vetores

#### Builders Sintáticos
- **DependencyParsingAdjacency**: Utiliza Stanford CoreNLP para análise de dependências
- **ConstituencyParsingAdjacency**: Baseado em árvores de constituintes
- **LiwcAdjacency**: Aplica Linguistic Inquiry and Word Count

#### Builders Baseados em Frequência
- **FrequencyAdjacency**: Implementa TF-IDF para ponderação documento-palavra

### 3.4 Camada de Modelos (Model Layer)

Implementa arquiteturas de GNN para classificação:

#### 3.4.1 Graph Convolutional Network (GCN)

O modelo GCN implementado segue a arquitetura de duas camadas proposta por Kipf e Welling [1]:

$$Z = \text{softmax}\left(\tilde{A} \text{ReLU}\left(\tilde{A}XW^{(0)}\right)W^{(1)}\right)$$

onde:
- $X$ é a matriz de características dos nós (inicialmente identidade para nós de palavras)
- $W^{(0)} \in \mathbb{R}^{F \times H}$ são os pesos da primeira camada
- $W^{(1)} \in \mathbb{R}^{H \times C}$ são os pesos da segunda camada
- $F$ é a dimensão de entrada, $H$ é a dimensão oculta, $C$ é o número de classes
- $\tilde{A} = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$ é a matriz de adjacência normalizada

#### 3.4.2 Normalização da Matriz de Adjacência

A normalização simétrica é aplicada para estabilizar o treinamento:

$$\tilde{A} = D^{-\frac{1}{2}}(A + I)D^{-\frac{1}{2}}$$

onde $D$ é a matriz diagonal de graus: $D_{ii} = \sum_j (A + I)_{ij}$.

### 3.5 Camada de Utilitários (Utils Layer)

Fornece funcionalidades auxiliares:
- **Device Manager**: Gerenciamento automático de dispositivos (CPU/GPU)
- **Logger**: Sistema de logging para rastreamento de processos
- **File Operations**: Operações de arquivo para persistência de dados

## 4. Metodologia de Implementação

### 4.1 Pipeline de Processamento

O pipeline completo do Text4GCN segue os seguintes passos:

1. **Carregamento do Dataset**: Download e carregamento dos dados brutos
2. **Pré-processamento**: Aplicação do `TextPipeline` para limpeza e tokenização
3. **Construção do Grafo**: Seleção e execução de um builder para gerar a matriz de adjacência
4. **Preparação dos Dados**: Carregamento da matriz de adjacência e características dos nós
5. **Treinamento**: Execução do modelo GCN com os dados preparados
6. **Avaliação**: Cálculo de métricas de desempenho (acurácia, F1-score, etc.)

### 4.2 Construção da Matriz de Adjacência

O processo de construção da matriz de adjacência heterogênea envolve:

1. **Extração do Vocabulário**: Identificação de todas as palavras únicas no corpus
2. **Construção de Relações Palavra-Palavra**: Aplicação do método selecionado (semântico, sintático ou frequência)
3. **Construção de Relações Documento-Palavra**: Cálculo de pesos TF-IDF
4. **Combinação**: Agregação das relações palavra-palavra e documento-palavra em uma única matriz esparsa

A matriz final tem dimensão $(N_d + N_w) \times (N_d + N_w)$, onde $N_d$ é o número de documentos e $N_w$ é o tamanho do vocabulário.

### 4.3 Treinamento do Modelo

O treinamento segue o seguinte procedimento:

1. **Divisão dos Dados**: Separação em conjuntos de treinamento, validação e teste
2. **Inicialização**: Configuração dos pesos da rede neural
3. **Forward Pass**: Propagação através das camadas GCN
4. **Cálculo da Perda**: Utilização de entropia cruzada:

$$\mathcal{L} = -\sum_{i \in \mathcal{Y}_L} \sum_{c=1}^{C} Y_{ic} \log Z_{ic}$$

onde $\mathcal{Y}_L$ é o conjunto de nós rotulados (documentos de treinamento), $Y$ são os rótulos verdadeiros e $Z$ são as predições.

5. **Backward Pass**: Atualização dos pesos via descida de gradiente estocástica
6. **Early Stopping**: Parada antecipada baseada na perda de validação

## 5. Características e Funcionalidades

### 5.1 Otimização de Memória

O Text4GCN utiliza matrizes esparsas (CSR - Compressed Sparse Row) para representar grafos grandes de forma eficiente em memória. Isso permite processar datasets com milhares de documentos e dezenas de milhares de palavras.

### 5.2 Flexibilidade de Configuração

O framework oferece múltiplos parâmetros configuráveis:
- **Threshold de Similaridade**: Para métodos baseados em similaridade
- **Tamanho da Janela**: Para métodos baseados em coocorrência
- **Dimensões Ocultas**: Tamanho das camadas ocultas do GCN
- **Taxa de Dropout**: Regularização durante o treinamento
- **Taxa de Aprendizado**: Controla a velocidade de convergência

### 5.3 Suporte a Múltiplos Dispositivos

O framework detecta automaticamente e utiliza GPUs quando disponíveis, com fallback automático para CPU. Isso é gerenciado pelo `DeviceManager`, que otimiza o uso de recursos computacionais.

## 6. Exemplos de Uso

### 6.1 Exemplo Básico com TF-IDF

```python
import text4gcn

# 1. Carregar dataset
data = text4gcn.datasets.R8(path="./data")

# 2. Construir grafo usando TF-IDF
builder = text4gcn.builder.Frequency()
adjacency = builder.tfidf(dataset_name="R8", dataset_path="./data")

# 3. Treinar modelo GCN
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

### 6.2 Exemplo com Word2Vec

```python
# Construir grafo usando Word2Vec
builder = text4gcn.builder.Embedding()
adjacency = builder.word2vec(
    dataset_name="R8",
    dataset_path="./data",
    training_regime="skipgram",
    embedding_dimension=200,
    num_epochs=10,
    threshold=0.95
)

# Treinar modelo
model = text4gcn.models.GNN(
    dataset="R8",
    path="./data",
    builder=text4gcn.builder.Builder.Embedding,
    layer=text4gcn.models.Layer.GCN
)
model.fit()
```

### 6.3 Exemplo com BERT

```python
# Construir grafo usando BERT
builder = text4gcn.builder.Semantic()
adjacency = builder.bert(
    dataset_name="R8",
    dataset_path="./data",
    language="english",
    threshold=0.95
)
```

## 7. Resultados e Análise

### 7.1 Datasets Utilizados

O framework foi avaliado em múltiplos datasets de benchmark:

| Dataset | Documentos | Treinamento | Teste | Vocabulário | Nós | Classes | Comprimento Médio |
|---------|------------|-------------|-------|-------------|-----|---------|-------------------|
| 20NG    | 18,846     | 11,314      | 7,532 | 42,757      | 61,603 | 20 | 221.26 |
| R8      | 7,674      | 5,485       | 2,189 | 7,688       | 15,362 | 8  | 65.72  |
| R52     | 9,100      | 6,532       | 2,568 | 8,892       | 17,992 | 52 | 69.82  |

### 7.2 Impacto da Filtragem Sintática e Semântica

A principal contribuição do Text4GCN é a capacidade de aplicar filtragem sintática e semântica que resulta em grafos menores e mais representativos. Isso traz os seguintes benefícios:

1. **Redução de Dimensionalidade**: Grafos menores requerem menos memória e computação
2. **Melhor Representação**: Conexões mais relevantes melhoram a qualidade das representações aprendidas
3. **Treinamento Mais Rápido**: Menos arestas aceleram a propagação de mensagens no GCN

### 7.3 Comparação de Métodos

Diferentes métodos de construção de adjacência capturam diferentes aspectos dos dados:

- **TF-IDF**: Captura relações estatísticas baseadas em frequência
- **Word2Vec/BERT**: Captura relações semânticas através de embeddings
- **Dependency Parsing**: Captura relações sintáticas estruturais

A escolha do método depende das características do dataset e do objetivo da aplicação.

## 8. Conclusão

Este artigo apresentou o **Text4GCN**, um framework Python de código aberto para classificação de texto utilizando Redes Neurais Convolucionais em Grafos. O framework oferece:

1. **Flexibilidade**: Múltiplos métodos para construção de grafos (semânticos, sintáticos e baseados em frequência)
2. **Eficiência**: Otimização de memória através de matrizes esparsas
3. **Facilidade de Uso**: API intuitiva que simplifica o pipeline completo
4. **Extensibilidade**: Arquitetura modular que permite adicionar novos métodos facilmente

A principal contribuição é fornecer uma ferramenta que combina técnicas avançadas de PLN com GCNs, permitindo a construção de grafos de texto otimizados que melhoram o desempenho em tarefas de classificação.

Trabalhos futuros incluem:
- Implementação de Graph Attention Networks (GAT)
- Suporte a mais datasets
- Otimizações adicionais para datasets muito grandes
- Integração com outros modelos de linguagem pré-treinados

## 9. Referências

[1] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *Proceedings of the 5th International Conference on Learning Representations (ICLR)*.

[2] Yao, L., Mao, C., & Luo, Y. (2018). Graph convolutional networks for text classification. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(01), 7370-7377.

[3] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*.

[4] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.

[5] Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)*, 1532-1543.

[6] Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. *arXiv preprint arXiv:1710.10903*.

[7] De Marneffe, M. C., & Manning, C. D. (2008). The Stanford typed dependencies representation. *Coling 2008: proceedings of the workshop on Cross-Framework and Cross-Domain Parser Evaluation*, 1-8.

[8] Pennebaker, J. W., Francis, M. E., & Booth, R. J. (2001). Linguistic inquiry and word count: LIWC 2001. *Mahway: Lawrence Erlbaum Associates*, 71(2001), 2001.

[9] Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information processing & management*, 24(5), 513-523.

[10] Text4GCN: A Python framework for text classification using Graph Neural Networks. Disponível em: https://github.com/vitormeriat/text4gcn

---

**Nota**: Este documento foi gerado com base na implementação do framework Text4GCN versão 2.0.0. Para mais informações, consulte a documentação oficial e o repositório do projeto.

