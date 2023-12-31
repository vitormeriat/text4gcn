#import modules.clean_data as cldt
#from ..modules import clean_data as clean
from text4gcn.modules import clean_data as clean
import time



class Metadata:
    '''
    Metadata class to define a decorator to save all the metadata related with the execution 
    of a function and save all to the database
    '''

    def metadata(key: str):
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
