# -*- coding: utf-8 -*-
"""
    Collection of miscellaneous utilities.
"""
import datetime, time, os

"""
Class for setting a minimum time between calls to a function. Useful for
ensuring that get requests and API calls don't hit too often.

Credit to ChrisTM on GitHub for putting this together.
URL: https://gist.github.com/ChrisTM/5834503

I have made some minor modifications to allow an arbitrary time increment,
and sleeping rather than suppressing requests.
"""
THROTTLE_VERBOSE = True
class throttle(object):
    """
    Decorator that prevents a function from being called more than once every
    time period.
    To create a function that cannot be called more than once a minute:
        @throttle(minutes=1)
        def my_fun():
            pass
    """
    def __init__(self, **kwargs):
        self.throttle_period = datetime.timedelta(**kwargs)
        self.time_of_last_call = datetime.datetime.now()

    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            now = datetime.datetime.now()
            time_since_last_call = now - self.time_of_last_call

            if time_since_last_call > self.throttle_period:
                self.time_of_last_call = now
                return fn(*args, **kwargs)
            else:
                sleep_dur = self.throttle_period - time_since_last_call
                if THROTTLE_VERBOSE:
                    print('Sleeping for ' + str(sleep_dur))
                time.sleep(sleep_dur.total_seconds())
                self.time_of_last_call = datetime.datetime.now()
                return fn(*args, **kwargs)
        return wrapper
    
def flatten_list(lst):
    """
        Collapse a list of lists
    """
    return [item for sublst in lst for item in sublst]


def list_contents(directory, ext='txt'):
    """
        List all the files in a directory with a given extension
    """
    return ['\\'.join([directory, file])
            for file in os.listdir(directory) if file.endswith('.'+ext)]