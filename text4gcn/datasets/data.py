import requests
import os


def list():
    return ['R8', 'R52', 'AG_NEWS']


class Datasets():
    def __init__(self, file_url: str, path: str, file_name: str) -> None:
        self.file_url = file_url
        self.path = path
        self.file_name = file_name

    def download(self):
        # URL of the image to be downloaded is defined as image_url
        r = requests.get(self.file_url)  # create HTTP response object

        #file_path = f'{os.getcwd()}/{path}'
        absp = os.path.abspath(self.path)
        file_path = f'{absp}/{self.file_name}'

        # send a HTTP request to the server and save
        # the HTTP response in a response object called r
        with open(file_path, 'wb') as f:

            # write the contents of the response (r.content)
            # to a new file in binary mode.
            f.write(r.content)

    def download_big(self):
        r = requests.get(self.file_url, stream=True)

        absp = os.path.abspath(self.path)
        file_path = f'{absp}/{self.file_name}'

        with open(file_path, "wb") as pdf:
            for chunk in r.iter_content(chunk_size=1024):

                # writing one chunk at a time to file
                if chunk:
                    pdf.write(chunk)


def R8(path: str):
    """R8 Dataset
    For additional details refer to http://ai.stanford.edu/~amaas/data/sentiment/
    Number of lines per split:
        - train: 25000
        - test: 25000
    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `test`)
    :returns: DataPipe that yields tuple of label (1 to 2) and text containing the movie review
    :rtype: (int, str)
    """
    ds = Datasets(
        file_url="https://meriatblog.blob.core.windows.net/public/text4gcn/R8.txt",
        file_name="R8.txt",
        path=path)
    ds.download()

    ds = Datasets(
        file_url="https://meriatblog.blob.core.windows.net/public/text4gcn/R8.meta",
        file_name="R8.meta",
        path=path)
    ds.download()


def R52(path: str):
    """R52 Dataset
    For additional details refer to http://ai.stanford.edu/~amaas/data/sentiment/
    Number of lines per split:
        - train: 25000
        - test: 25000
    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `test`)
    :returns: DataPipe that yields tuple of label (1 to 2) and text containing the movie review
    :rtype: (int, str)
    """
    ds = Datasets(
        file_url="https://meriatblog.blob.core.windows.net/public/text4gcn/R52.txt",
        file_name="R52.txt",
        path=path)
    ds.download()

    ds = Datasets(
        file_url="https://meriatblog.blob.core.windows.net/public/text4gcn/R52.meta",
        file_name="R52.meta",
        path=path)
    ds.download()


def AG_NEWS(path: str):
    """AG_NEWS Dataset
    For additional details refer to http://ai.stanford.edu/~amaas/data/sentiment/
    Number of lines per split:
        - train: 25000
        - test: 25000
    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `test`)
    :returns: DataPipe that yields tuple of label (1 to 2) and text containing the movie review
    :rtype: (int, str)
    """
    ds = Datasets(
        file_url="https://meriatblog.blob.core.windows.net/public/text4gcn/20AG_NEWS.txt",
        file_name="20AG_NEWS.txt",
        path=path)
    ds.download()

    ds = Datasets(
        file_url="https://meriatblog.blob.core.windows.net/public/text4gcn/20AG_NEWS.meta",
        file_name="20AG_NEWS.meta",
        path=path)
    ds.download()
