#!/usr/bin/python2.7
 
import os
import sys
import shutil
 
def convert_to_utf8(filename):
    # gather the encodings you think that the file may be
    # encoded inside a tuple
    encodings = 'iso-8859-15'
 
    # try to open the file and exit if some IOError occurs
    try:
        f = open(filename, 'r').read()
    except Exception:
        sys.exit(1)
 
    data = f.decode(encodings)
 
    # now get the absolute path of our filename and append .bak
    # to the end of it (for our backup file)
    fpath = os.path.abspath(filename)
    newfilename = fpath + '.bak'
    # and make our backup file with shutil
    shutil.copy(filename, newfilename)
 
    # and at last convert it to utf-8
    f = open(filename, 'w')
    try:
        f.write(data.encode('utf-8'))
    except Exception, e:
        print e
    finally:
        f.close()