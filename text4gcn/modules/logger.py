#from ..modules import clean_data as clean
from text4gcn.modules import clean_data as clean
import datetime
import time


INFO = "[INFO]"
ERROR = "[ERROR]"
WARNING = "[WARN]"


class PrintLog():

    def __init__(self):
        self.output = []

    def log(self, msg='', end='\n'):
        now = datetime.datetime.now()
        t = (
            (
                (
                    f'{str(now.year)}/{str(now.month)}/{str(now.day)} '
                    + str(now.hour).zfill(2)
                )
                + ':'
            )
            + str(now.minute).zfill(2)
            + ':'
        ) + str(now.second).zfill(2)

        lines = msg.split('\n') if isinstance(msg, str) else [msg]

        for line in lines:
            if line == lines[-1]:
                print(f'[{t}] {str(line)}', end=end)
            else:
                print(f'[{t}] {str(line)}')
            self.output.append(str(line)+end)

    def info(self, msg='', end='\n'):
        self.log(msg=f'{INFO} {msg}', end=end)

    def warning(self, msg='', end='\n'):
        self.log(msg=f'{WARNING} {msg}', end=end)

    def error(self, msg='', end='\n'):
        self.log(msg=f'{ERROR} {msg}', end=end)

    def log_history(self):
        return self.output


class Process:
    '''
    Metadata class to define a decorator to save all the metadata related with the execution 
    of a function and save all to the database
    '''

    def log(key: str):
        '''
        function created to decorate another funcion
        :param key: name of the function (entry in the database)
        '''

        def metadata_decor(func: callable):
            '''
            decorator
            :param func: call function
            '''
            def wrapper(*args, **kwargs):
                '''
                wrapper
                '''
                args[0].logger.info(clean.create_title(key))

                initTime = time.time()
                args[0].logger.info(f"Start process: {initTime}")

                val = func(*args, **kwargs)

                endTime = time.time()
                totalTime = endTime-initTime

                args[0].logger.info("Elapsed time is %f seconds." % totalTime)
                args[0].logger.info()

                return val

            return wrapper

        return metadata_decor
