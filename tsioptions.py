#! /usr/bin/env python3
# =*- ociding: utf-8 -*-
#
# tsioptions.py - a very simple argument/option handler
#
# Copyright (c) 2016, Phil Maker
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of the copyright-owner nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

'''TSI option/argv handler

This is intended to be the simplest possible option/argv
imaginable. Comand line arguments are always in pairs and a dictionary
is used to keep the values.
'''

import sys

# argv handling
options = {}
'''dictionary of command line options'''


def set(defaults):
    '''set default values and process argv'''
    global options
    options = defaults.copy()

    # Note: sphinx-build passes its own arguments so ignore it
    if sys.argv[0].endswith('sphinx-build'):
        return

    # update options
    if len(sys.argv[1:]) > 0:
        options.update([sys.argv[1:]])
        for o in options:
            if not (o in defaults):
                print('fatal error: option',
                      o, 'not defined arguments: see defaults')
                for o in defaults:
                    print(o, defaults[o])
                exit(101)

    # finally display them
    if options['-show_options']:
        show()


def get(k):
    '''return the value for an option'''
    global options
    return options[k]


def show():
    '''print out the current options and defauts'''
    for o in sorted(options):
        print('options ' + o.ljust(24) + ' ' +
              str(options[o]))


# finally call main (or profile it)
if __name__ == '__main__':
    main()
