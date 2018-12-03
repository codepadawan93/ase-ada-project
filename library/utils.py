'''
Utility functions go here. All are static
'''

class Utils:

    logfile = "datalysis.log"

    # Make sure someone doesn't create Utils objects
    def __init__(self):
        raise TypeError("Utils is non-instantiatable.")

    @staticmethod
    def log(tag, message, file=logfile):
        file = open(file, "w+")
        file.write(f"${tag}:${message}\r\n")
        file.close()

    @staticmethod
    def putfile(filename, contents):
        file = open(filename, "w+")
        file.write(contents)
        file.close()



