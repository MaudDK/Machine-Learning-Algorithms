from timeit import default_timer as timer
import numpy as np

def measure_speed(func):
    def wrapper(*args, **kwargs):
        start_time = timer()

        result = func(*args, **kwargs)

        end_time = timer()

        exec_time = end_time - start_time
        
        print("Execution time of", func.__name__, ":", exec_time * 1000 , "ms")
        return result
    
    return wrapper