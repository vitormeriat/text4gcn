from text4gcn.builder.cosine_similarity_adj import CosineSimilarityAdjacency
from text4gcn.builder.dependency_parsing_adj import DependencyParsingAdjacency
from text4gcn.builder.word2vec_adj import Word2VecAdjacency
from text4gcn.builder.frequency_adj import FrequencyAdjacency
from text4gcn.builder.linguistic_inquiry_adj import LiwcAdjacency
from text4gcn.builder.bert_adj import BERTAdjacency



class Adjacency():
    def __init__(self, dataset_name: str, dataset_path: str) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path


class Frequency(Adjacency):
    def __init__(self, dataset_name: str, dataset_path: str) -> None:
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path)

    def tfidf(self, window_size: int = 20, output_path: str = None) -> bool:
        adj = FrequencyAdjacency(dataset_name=self.dataset_name,
                                 dataset_path=self.dataset_path,
                                 window_size=window_size,
                                 output_path=output_path)
        return adj.build()


class Syntactic(Adjacency):
    def __init__(self, dataset_name: str, dataset_path: str) -> None:
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path)

    def dependency_parsing(self, core_nlp_path: str, language: str = "english") -> bool:
        """
        Working in Progress
        ----------

        """
        adj = DependencyParsingAdjacency(dataset_name=self.dataset_name,
                                         dataset_path=self.dataset_path,
                                         core_nlp_path=core_nlp_path,
                                         language=language)
        return adj.build()

    def constituency_parsing(self) -> bool:
        """
        Working in Progress
        ----------

        """
        pass

    def linguistic_inquiry(self, liwc_path: str) -> bool:
        """
        Working in Progress
        ----------

        """
        adj = LiwcAdjacency(dataset_name=self.dataset_name,
                            dataset_path=self.dataset_path,
                            liwc_path=liwc_path)
        return adj.build()


class Semantic(Adjacency):
    def __init__(self, dataset_name: str, dataset_path: str) -> None:
        super().__init__(dataset_name=dataset_name, dataset_path=dataset_path)

    def word2vec(self,
                 training_algorithm: int,
                 embedding_dimension: int,
                 num_epochs: int,
                 language: str = "english",
                 threshold: float = 0.95,
                 output_path: str = None) -> bool:
        """
        Create a folder structure to store the files representing the 
        processed text and the node resources to be used to create the 
        text graph.

        ...

        Attributes
        ----------
        dataset_name : str
            name of the dataset
        dataset_path : str
            path to files
        language : str
            `language` to retrieve stop words
        training_algorithm : int
            1 to CBOW and 2 to SKIP GRAM
        embedding_dimension : int
            Embedding Dimensions
        num_epochs : int
            Number of epochs used to train the Word2Vec
        threshold : float
            Treshold to apply the filter based on cosine similarity
        output_path : str
            Output path optional to create the ajacency files
        """
        adj = Word2VecAdjacency(dataset_name=self.dataset_name,
                                dataset_path=self.dataset_path,
                                training_regime=training_algorithm,
                                embedding_dimension=embedding_dimension,
                                num_epochs=num_epochs,
                                language=language,
                                threshold=threshold,
                                output_path=output_path)
        return adj.build()

    def bert(self, 
             language: str = "english",
             threshold: float = 0.95) -> bool:
        """
        Working in Progress
        ----------

        """
        adj = BERTAdjacency(dataset_name=self.dataset_name, 
                            dataset_path=self.dataset_path, 
                            language=language, 
                            threshold=threshold)
        return adj.build()

    def llm(self) -> bool:
        """
        Working in Progress
        ----------

        """
        pass

    def cosine_similarity(self,
                          threshold: float = 0.95,
                          output_path: str = None) -> bool:
        """
        Create a folder structure to store the files representing the 
        processed text and the node resources to be used to create the 
        text graph.

        ...

        Attributes
        ----------
        dataset_name : str
            name of the dataset
        dataset_path : str
            path to files
        threshold : float
            Treshold to apply the filter based on cosine similarity
        output_path : str
            Output path optional to create the ajacency files
        """
        adj = CosineSimilarityAdjacency(dataset_path=self.dataset_path,
                                        dataset_name=self.dataset_name,
                                        threshold=threshold,
                                        output_path=output_path)
        return adj.build()
