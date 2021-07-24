import time
from functools import wraps


def timing(f):
    """Decorator for timing functions
    Usage:
    @timing
    def function(a):
        pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print("function:%r took: %2.5f sec" % (f.__name__, end - start))
        return result

    return wrapper
