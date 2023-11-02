import os
import sys
import copy


class Config:
    """
    Simple configuration object to house named arguments for experiments

    File format is:
    # Comments start with pound symbol
    arg_name = arg_value
    arg_name = arg_value # side comment that will be stripped off

    @author: Alex Ororbia
    """

    def __init__(self, fname):
        self.fname = fname
        self.variables = {}

        fd = open(fname, 'r')
        count = 0
        while True:
            count += 1
            line = fd.readline()
            if not line:
                break
            line = line.replace(" ", "").replace("\n", "")
            if len(line) > 0:
                cmt_split = line.split("#")
                argmt = cmt_split[0]
                if (len(argmt) > 0) and ("=" in argmt):
                    tok = argmt.split("=")
                    var_name = tok[0]
                    var_val = tok[1]
                    self.variables[var_name] = var_val
        fd.close()

    def getArg(self, arg_name):
        """
            Retrieve argument from current configuration
        """
        return self.variables.get(arg_name)

    def hasArg(self, arg_name):
        """
            Check if argument exists (or if it is known by this config object)
        """
        arg = self.variables.get(arg_name)
        flag = False
        if arg is not None:
            flag = True
        return flag
