# from text4gcn.modules.textvec.word_features import train_word2vec
from text4gcn.modules import adjacency as adjcy
from text4gcn.modules import logger as logger
from text4gcn.modules import file_ops as flop
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
import pickle
from tqdm import tqdm


from sklearn.metrics.pairwise import cosine_similarity as cos

# Lazily loaded — importing text4gcn.builder must not hit HuggingFace.
_tokenizer = None
_model = None


def _get_bert():
    """Load BERT tokenizer/model on first use (requires transformers + network/cache)."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        try:
            from transformers import BertModel, BertTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for Semantic.bert().\n"
                "Install with: pip install 'text4gcn[bert]'"
            ) from exc

        _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _model = BertModel.from_pretrained("bert-base-uncased")
    return _tokenizer, _model


def get_word_embedding(word: str, model, tokenizer):
    import torch

    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Pegando o primeiro token depois do [CLS]
    embedding = outputs.last_hidden_state[0][1].numpy()
    return embedding


def similarity(vec1, vec2) -> float:
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cos(vec1, vec2)[0][0]


def get_similarity(word1, word2, model, tokenizer):
    embedding1 = get_word_embedding(word1, model, tokenizer)
    embedding2 = get_word_embedding(word2, model, tokenizer)
    return similarity(embedding1, embedding2)


def get_embeddings(words_batch):
    """Extract CLS embeddings for a batch of words (loads BERT on first call)."""
    import torch

    tokenizer, model = _get_bert()
    inputs = tokenizer(words_batch, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy()


class BERTAdjacency():

    def __init__(self,
                 dataset_name,
                 dataset_path,
                 # training_regime,
                 # embedding_dimension,
                 # num_epochs,
                 language: str = "english",
                 threshold: float = 0.95,
                 # output_path: str = None
                 ):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        # self.training_regime = training_regime
        # self.embedding_dimension = embedding_dimension
        # self.num_epochs = num_epochs
        self.logger = logger.PrintLog()
        self.flop = flop.FileOps(logger=self.logger)
        self.language = language
        self.threshold = threshold
        # self.output_path = output_path

    def build_relation_pair(self, docs_of_words):
        stop_words = set(stopwords.words(self.language))

        errors = 0
        rela_pair_count_str = {}

        chunk_size=10
        result=[docs_of_words[i:i + chunk_size] for i in range(0, len(docs_of_words), chunk_size)]
        print(f'[INFO] LEN: {len(result)}')

        for r in tqdm(result):
            

            docs_of_word = []
            for i in r:
                docs_of_word.extend(i)

            docs_of_word_len = len(docs_of_word)

            embeddings = get_embeddings(docs_of_word)

            similarity_matrix = cos(embeddings)

        
        # for docs_of_word in tqdm(docs_of_words):
        #     docs_of_word_len = len(docs_of_word)

        #     # TODO:
        #     # Dividir a lista de palavras em lotes
        #     batch_size = 32 * 16  # Ajuste o tamanho do lote conforme necessário
        #     batches = [docs_of_word[i:i + batch_size] for i in range(0, len(docs_of_word), batch_size)]

        #     # Extrair embeddings para todas as palavras
        #     all_embeddings = []

        #     for batch in batches:
        #         embeddings = get_embeddings(batch)
        #         all_embeddings.extend(embeddings)

        #     all_embeddings = np.array(all_embeddings)

        #     # Calcular a matriz de similaridade
        #     similarity_matrix = cos(all_embeddings)

            for i in range(docs_of_word_len):
                if i+1 == docs_of_word_len:
                    continue

                if docs_of_word[i] == docs_of_word[i+1]:
                    continue

                if docs_of_word[i] in stop_words or docs_of_word[i+1] in stop_words:
                    continue

                # cosine_similarity = get_similarity(word1=docs_of_word[i],
                #                                    word2=docs_of_word[i+1],
                #                                    model=model,
                #                                    tokenizer=tokenizer)

                # Exemplo de como acessar a similaridade entre duas palavras específicas
                word_index_1 = docs_of_word.index(docs_of_word[i])
                word_index_2 = docs_of_word.index(docs_of_word[i+1])
                cosine_similarity = similarity_matrix[word_index_1, word_index_2]

                if cosine_similarity < self.threshold:
                    errors += 1
                    continue

                word_pair_str = f'{docs_of_word[i]},{docs_of_word[i+1]}'
                if word_pair_str in rela_pair_count_str:
                    rela_pair_count_str[word_pair_str] += 1
                else:
                    rela_pair_count_str[word_pair_str] = 1
                # two orders
                word_pair_str = f'{docs_of_word[i+1]},{docs_of_word[i]}'
                if word_pair_str in rela_pair_count_str:
                    rela_pair_count_str[word_pair_str] += 1
                else:
                    rela_pair_count_str[word_pair_str] = 1

        print(f'[INFO] Words without similarity: {errors}')
        return rela_pair_count_str

    # def build_relation_pair(self, docs_of_words):
    #     stop_words = set(stopwords.words(self.language))

    #     errors = 0
    #     rela_pair_count_str = {}
    #     for docs_of_word in tqdm(docs_of_words):
    #         docs_of_word_len = len(docs_of_word)

    #         # TODO:
    #         # Dividir a lista de palavras em lotes
    #         batch_size = 32 * 16  # Ajuste o tamanho do lote conforme necessário
    #         batches = [docs_of_word[i:i + batch_size] for i in range(0, len(docs_of_word), batch_size)]

    #         # Extrair embeddings para todas as palavras
    #         all_embeddings = []

    #         for batch in batches:
    #             embeddings = get_embeddings(batch)
    #             all_embeddings.extend(embeddings)

    #         all_embeddings = np.array(all_embeddings)

    #         # Calcular a matriz de similaridade
    #         similarity_matrix = cos(all_embeddings)

    #         for i in range(docs_of_word_len):
    #             if i+1 == docs_of_word_len:
    #                 continue

    #             if docs_of_word[i] == docs_of_word[i+1]:
    #                 continue

    #             if docs_of_word[i] in stop_words or docs_of_word[i+1] in stop_words:
    #                 continue

    #             # cosine_similarity = get_similarity(word1=docs_of_word[i],
    #             #                                    word2=docs_of_word[i+1],
    #             #                                    model=model,
    #             #                                    tokenizer=tokenizer)

    #             # Exemplo de como acessar a similaridade entre duas palavras específicas
    #             word_index_1 = docs_of_word.index(docs_of_word[i])
    #             word_index_2 = docs_of_word.index(docs_of_word[i+1])
    #             cosine_similarity = similarity_matrix[word_index_1, word_index_2]

    #             if cosine_similarity < self.threshold:
    #                 errors += 1
    #                 continue

    #             word_pair_str = f'{docs_of_word[i]},{docs_of_word[i+1]}'
    #             if word_pair_str in rela_pair_count_str:
    #                 rela_pair_count_str[word_pair_str] += 1
    #             else:
    #                 rela_pair_count_str[word_pair_str] = 1
    #             # two orders
    #             word_pair_str = f'{docs_of_word[i+1]},{docs_of_word[i]}'
    #             if word_pair_str in rela_pair_count_str:
    #                 rela_pair_count_str[word_pair_str] += 1
    #             else:
    #                 rela_pair_count_str[word_pair_str] = 1

    #     print(f'[INFO] Words without similarity: {errors}')
    #     return rela_pair_count_str

    def compute_weights(self, docs_of_words, rela_pair_count_str, word_window_freq, train_size, word_id_map, vocab_size, word_doc_freq, vocab):
        row = []
        col = []

        # compute weights PMI
        weight = adjcy.compute_weights_with_PMI(docs_of_words,
                                                rela_pair_count_str,
                                                word_window_freq,
                                                train_size,
                                                word_id_map,
                                                row,
                                                col)

        # doc word frequency
        doc_word_freq = adjcy.get_doc_word_freq(docs_of_words, word_id_map)

        weight_tfidf = adjcy.get_weight_tfidf(docs_of_words,
                                              word_id_map,
                                              doc_word_freq,
                                              train_size,
                                              vocab_size,
                                              word_doc_freq,
                                              vocab,
                                              row,
                                              col)

        return row, col, weight, weight_tfidf

    def _save_history(self, hist):
        self.flop.create_dir(
            dir_path=f'{self.dataset_path}/log', overwrite=False)
        with open(f'{self.dataset_path}/log/{self.dataset_name}_dataset.txt', 'a', encoding="utf-8") as my_file:
            my_file.writelines(hist)

    def build(self):

        corpus_path = f"{self.dataset_path}/{self.dataset_name}"
        ds_corpus = f'{corpus_path}.shuffled/{self.dataset_name}.txt'
        ds_corpus_vocabulary = f'{corpus_path}.shuffled/{self.dataset_name}.vocab'
        ds_corpus_train_idx = f'{corpus_path}.shuffled/{self.dataset_name}.train'
        ds_corpus_test_idx = f'{corpus_path}.shuffled/{self.dataset_name}.test'

        # checkers
        # check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
        # check_paths(ds_corpus, ds_corpus_vocabulary, ds_corpus_train_idx, ds_corpus_test_idx)
        # create_dir(dir_path=cfg.corpus_shuffled_adjacency_dir + "/semantic", overwrite=False)
        self.flop.create_dir(dir_path=f'{corpus_path}.adjacency',
                             overwrite=False)

        docs_of_words = [line.split() for line in open(file=ds_corpus, encoding="utf-8")]
        # Extract Vocabulary.
        vocab = open(ds_corpus_vocabulary, encoding="utf-8").read().splitlines()
        # Real train-size, not adjusted.
        train_size = len(open(ds_corpus_train_idx, encoding="utf-8").readlines())
        # Real test-size.
        test_size = len(open(ds_corpus_test_idx, encoding="utf-8").readlines())

        word_doc_freq = adjcy.define_word_doc_freq(docs_of_words)

        vocab_size = len(vocab)
        word_id_map = {}
        id_word_map = {}
        for i in range(vocab_size):
            word_id_map[vocab[i]] = i
            id_word_map[i] = vocab[i]

        word_window_freq = adjcy.build_word_window_freq(docs_of_words)

        rela_pair_count_str = self.build_relation_pair(docs_of_words)

        row, col, weight, weight_tfidf = self.compute_weights(docs_of_words,
                                                              rela_pair_count_str,
                                                              word_window_freq,
                                                              train_size,
                                                              word_id_map,
                                                              vocab_size,
                                                              word_doc_freq,
                                                              vocab)

        # =============================================================
        weight += weight_tfidf
        node_size = train_size + vocab_size + test_size

        self.logger.info(
            f"({len(weight)}, ({len(row)}, {len(col)})), shape=({node_size}, {node_size})")

        adj = csr_matrix((weight, (row, col)), shape=(node_size, node_size))
        # =============================================================
        # Dump Adjacency Matrix
        with open(f"{corpus_path}.adjacency/ind.embedding.bert.{self.dataset_name}.adj", 'wb') as f:
            pickle.dump(adj, f)

        hist = self.logger.log_history()
        self._save_history(hist)
