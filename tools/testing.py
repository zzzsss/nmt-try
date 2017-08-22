#!/bin/env python
#-*- encoding=utf8 -*-

import os,sys

if __name__=="__main__":
    print ("__file__=%s" % __file__)
    print ("os.path.realpath(__file__)=%s" % os.path.realpath(__file__))
    print ("os.path.dirname(os.path.realpath(__file__))=%s" % os.path.dirname(os.path.realpath(__file__)))
    print ("os.path.split(os.path.realpath(__file__))=%s" % os.path.split(os.path.realpath(__file__))[0])
    print ("os.path.abspath(__file__)=%s" % os.path.abspath(__file__))
    print ("os.getcwd()=%s" % os.getcwd())
    print ("sys.path[0]=%s" % sys.path[0])
    print ("sys.argv[0]=%s" % sys.argv[0])

    Traceback (most recent call last):
      File "/home/z/tmp/dynet/setup.py", line 113, in <module>
        append_cmake_lib_list(GPULIBRARIES, ENV.get("CUDA_CUBLAS_FILES"))
      File "/home/z/tmp/dynet/setup.py", line 34, in append_cmake_lib_list
        l.extend(map(strip_lib, var.split(";")))
      File "/home/z/tmp/dynet/setup.py", line 39, in strip_lib
        return re.sub(r"^(?:(?:lib)?(.*)\.(?:so|a|dylib)|(.*)\.lib)$", r"\1\2", filename)
      File "/usr/lib/python2.7/re.py", line 151, in sub
        return _compile(pattern, flags).sub(repl, string, count)
      File "/usr/lib/python2.7/re.py", line 278, in filter
        return sre_parse.expand_template(template, match)
      File "/usr/lib/python2.7/sre_parse.py", line 799, in expand_template
        raise error, "unmatched group"
    sre_constants.error: unmatched group
