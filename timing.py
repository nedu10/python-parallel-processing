import time

def timing(func):
    '''Used to time functions.'''
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start}'s")
        return ret
    return wrapper