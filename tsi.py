#! /usr/bin/env python3
#
# tsi.py - time series irregular library
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

'''tsi

This module....

Example:
   Example 1 with literal block::
      $ echo hello

TODO:
   #. Break this apart into individual components
'''



import datetime
import time
import calendar
import iso8601
import math
import statistics
import glob
import os
import sys
import pytz

import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tsioptions as opts

opts.set({
    '-trace': False,
    '-profile_main': False,
    '-show_options': True,
    '-test': 1
})

# option setup
plt.style.use('ggplot')
pd.options.display.width = 500

# timestamp support
def tparse(t):
    '''convert a ISO8601 timestamp to a float of seconds since UNIX epoch.

    Args:
      t (str): ISO8601 timestamp

    Returns:
      timeStamp: if t is correct, otherwise fails

    Examples:
      >>> import tsi
      >>> print(tsi.tparse('2001-09-09T01:46:40+00:00'))
      1000000000.0

      >>> import tsi
      >>> print(tsi.tparse('2001-09-09T01:46:40Z'))
      1000000000.0

      >>> import tsi
      >>> print(tsi.tparse('20010909T014640Z'))
      1000000000.0

      >>> import tsi
      >>> print(tsi.tparse('2001-09-09T01:46:40.123+00:00'))
      1000000000.123

      >>> import tsi
      >>> print(tsi.tparse('2001-09-09 01:46:40.123+00:00'))
      1000000000.123

      >>> import tsi
      >>> print(tsi.tparse('2001-09-09 01:46:40.978612+0000'))
      1000000000.978612

      >>> import tsi
      >>> print(tsi.tparse('2099-09-09 01:46:40.978612+0000'))
      4092601600.978612
    '''
    return iso8601.parse_date(t).timestamp()
    # return np.datetime64(t)

def tformat(s):
    '''convert a float of seconds since epoch to an ISO8601 date

    There is still a bit of representation error in this
    component. Why we aren't using numpy datetime64.

    Args:
      s (float): seconds since UNIX epoch.

    Returns:
      str: formatted time

    Examples:
      >>> import tsi
      >>> print(tsi.tformat(1000000000.0))
      2001-09-09T01:46:40+00:00

      >>> import tsi
      >>> print(tsi.tformat(1000000000.988))
      2001-09-09T01:46:40.988000+00:00

      Note that we get representation errors in the following
      two tests.

      !>>> import tsi
      !>>> print(tsi.tformat(1000000000.989))
      2001-09-09T01:46:40.989000+00:00

      !>>> import tsi
      !>>> print(tsi.tformat(tsi.tparse('2001-09-09T01:46:40.989000+00:00')))
      2001-09-09T01:46:40.989000+00:00
    '''
    return datetime.datetime.fromtimestamp(round(s, 3),
                                           pytz.UTC).isoformat('T')

def tdsecs(td):
    '''convert a timedelta to a number in seconds

    Args:
      ts (timedelta): a timedelta from pandas/numpy

    Returns:
      float: representation in seconds of delta

    Examples:
       None

    '''

    return (td / np.timedelta64(1, 's'))

# support for nan
from math import isnan
nan = float('nan')
'''float: just nan for us to use'''

# support for reading data in

hists = {}
'''{var->[(when, what)]}: history of var as a list of (when, what) events

This is used as the basic representation for history of variables
before we mangle it into the various pandas DataFrame representations.
'''

def tsvars():
    '''returns variables we have history for

    Returns:
    [str]: sorted list of variable names
    '''
    return sorted(hists)

def tsread(fn):
    '''read file fn and convert [(when, what)...]

    Args:
      fn (str): filename to read which is in ASIM format

    Returns:
      [(when,what)]: list of when, what events
    '''
    print('tsread ' + fn)
    n = 1
    r = []
    for s in open(fn):
        t, v = s.split(',')
        if n == 1:  # skip the header line
            assert t == 't'
            # v is whatever,we don't check
        else:
            t = tparse(t)
            v = float(v)
            r.append((t, v))
        n += 1
    return r

def tsreadfiles(pat, rename):
    '''Read all files matching glob pat and rename var using rename

    Args:

    pat (str): glob pattern for matching files
    rename (hook): function renaming variables from file to varname

    Returns:
    nothing
    '''
    global hists
    for fn in glob.glob(pat):
        var = rename(fn)
        hists[var] = tsread(fn)

def tsevents():
    '''converts hists[] to [(when, var, what)...]

    Returns:
    [(when, var, what)]: similar to an alarm log
    '''
    global hists
    events = []
    for var in sorted(hists):
        for (when, what) in hists[var]:
            events.append((when, var, what))
    events.sort()
    # print('tsevents = ', events)
    return events

def tsstates():
    '''converts hists to [when, {var->what}] by remembering state

    For example [a:[(100, 1), (200, 10)], b:[(10, -1), (250, -11)]]
    converts to
    [(10, {a:nan, b:-1}),
    (100, {a:1, b:-1}),
    (200, {a:10, b:-1}),
    (250, {a:10, b:-11})]

    Examples:

    None yet till I redo it

    Returns:
    [when, {var->what}]: it expands the events into states
    '''
    global hists
    states = []
    state = {}
    for v in hists:
        state[v] = nan
    lastwhen = nan
    for (when, var, what) in tsevents():
        state[var] = what
        if when != lastwhen:
            states.append((when, state.copy()))
    return states

def ts2csv(fd):
    '''print ts stat to fd'''
    ts2csvheader(fd)
    ts2csvbody(fd)

def ts2csvheader(fd):
    '''print the header line'''
    print('t', end=',', file=fd)
    for i in tsvars():
        print(i, end=',', file=fd)
    print('Remarks', file=fd)

def ts2csvbody(fd):
    '''print the body'''
    for (when, state) in tsstates():
        print(tformat(when), end=',', file=fd)
        for v in tsvars():
            print(state[v], end=',', file=fd)
        print('', file=fd)

def limit(v, low, high):
    '''limit v between low and high'''
    return max(min(v, high), low)

def scale(v, p=1):
    '''scale v by p'''
    return v*p

def offset(v, o=0):
    '''offset v by o'''
    return v+o

def fntovar(fn):
    '''Convert filename to variable name'''
    fn = fn.replace('data/', '')
    fn = fn.replace('R_K_PG_BULM_', '')
    fn = fn.replace('_1JAN2000_now', '')
    fn = fn.replace('.csv', '')
    return fn

def tsmean(df, v):
    '''Return the time weighted average for v in DataFrame df'''
    return df['w' + v].sum()/df['t' + v].sum()

def tssummary(df, v):
    '''Return a summary of the variable v in dataframe df.'''
    s = (v + ' min..tsmean..max = ').ljust(32)
    s += str(df[v].min()) + '..'
    s += str(round(tsmean(df, v), 3)) + '..'
    s += str(df[v].max())
    return s

def getdf(pats):
    '''Return DataFrame from files matching members of pats.

    pats - list of glob style pattern matching the files to process.
    '''
    for pat in pats:
        tsreadfiles(pat, fntovar)
    ts2csv(open('tmpdata.csv', 'w'))
    # pylint does not seem to understand what a dataframe is
    # whence we need to disable the check
    # pylint:disable=maybe-no-member
    return pd.read_csv('tmpdata.csv', parse_dates=['t']).set_index('t')

def makedt(df):
    '''Return a dataframe with new dt, w* and t* Series.

    The new dataframe contains:

    df['dt'] - the difference in time between this samples.
    df['w' + var] - the time weighted value for var.
    df['t' + var] - the dt for var if it is not nan otherwise 0
    '''
    print('makedt')
    df['tvalue'] = df.index
    df['dt'] = (df['tvalue']-df['tvalue'].shift()).fillna(0).shift(-1)
    df['dt'] = df['dt'].apply(tdsecs)
    del df['tvalue']
    f = lambda x: 0 if isnan(x[0]) else x[1]
    for v in tsvars():
        df['w' + v] = df[v] * df['dt']
        df['t' + v] = df[[v, 'dt']].apply(f, axis=1)
    return df

# wrappers for plot

from matplotlib.backends.backend_pdf import PdfPages

def plotPdf(fn, **kwopts):
    print('plotPdf ' + fn)
    pp = PdfPages(fn)
    year = 2015
    months = range(10, 13)
    for month in months:
        dmax = calendar.monthrange(year, month)[1]
        for day in range(1, dmax-1):
            title = str(year) + '-' + str(month) + '-' + str(day)
            print(' plotPdf ' +  title)
            global df
            for v in tsvars():
                df[v].plot(kind='line',
                           # color='black',
                           drawstyle='steps',
                           title=title,
                           xlim=(pd.datetime(year, month, day, 19),
                                 pd.datetime(year, month, day+1, 10)))
            pp.savefig()
    pp.close()
    os.system('evince ' + fn)

# trace code
#
# TODO: update to a better tracer sometime
#

if opts.get('-trace'):
    import sys

    def tracer(frame, event, arg, indent=[0]):
        # print('tracer', event)
        func = frame.f_code.co_name
        file = frame.f_code.co_filename
        if func[0] == '_' or file[0] == '/':
            return
        if event == "call":
            indent[0] += 2
            print("-" * indent[0] + "> call function",
                  func + ':' + file)
        elif event == "return":
            print("<" + "-" * indent[0], "exit function",
                  func + ':' + file)
        indent[0] -= 2
        return tracer

    sys.setprofile(tracer)

# profile support
import cProfile
import pstats

def profile(c):
    '''profile code c
    Args:
    c (str): command to profile
    '''
    cProfile.run('main()', 'tm-stats')
    p = pstats.Stats('tm-stats')
    p.sort_stats('cumulative').print_stats(20)

# some 1 liners for testing
def showeval(s):
    '''print a string followed by its evaluation

    Note that python does not have an uplevel(3tcl) command
    like TCL and so need to use globals or the locals=dict
    argument. There must be a better way.
    '''
    print(s, '->', eval(s))

def main():
    '''Do the work'''
    test1_2()

def test1():
    '''run a simple test

    Note
    '''
    print('* test1() - basic input and statistics for 1 series')
    global df  # we need global dataframe so showeval can see it.
    df = {}
    print('** read data/Test1.csv see contents below')
    print(open('data/Test1.csv').read())
    print("df = makedt(getdf(['data/Test1.csv']))")
    df = makedt(getdf(['data/Test1.csv']))
    showeval("print(df)")
    showeval("print(df.describe())")
    print('** statistics - N.B. weighting seems odd')
    showeval("df['Test1'].mean() # wrongas expected")
    showeval("df['Test1'].mean(weighted=df['dt']) # wrong")
    showeval("df['Test1'].mean(weighted=df['tTest1']) # wrong")
    showeval("(df['Test1']*df['tTest1']).sum()/df['tTest1'].sum() # ok")
    showeval("tsmean(df,'Test1')")

def test1_2():
    '''run a simple test

    Note
    '''
    print('* test1_2() - basic input and statistics for 2 series')
    global df  # we need global dataframe so showeval can see it
    df = {}
    print('** read data/Test1.csv and data/Test2.csv see contents below')
    print(open('data/Test1.csv').read())
    print(open('data/Test2.csv').read())
    print("df = makedt(getdf(['data/Test1.csv','data/Test2.csv']))")
    df = makedt(getdf(['data/Test1.csv', 'data/Test2.csv']))
    showeval("print(df)")
    showeval("print(df.describe())")
    print('** statistics - N.B. weighting seems odd')
    showeval("df['Test1'].mean() # wrongas expected")
    showeval("df['Test1'].mean(weighted=df['dt']) # wrong")
    showeval("df['Test1'].mean(weighted=df['tTest1']) # wrong")
    showeval("(df['Test1']*df['tTest1']).sum()/df['tTest1'].sum() # ok")
    showeval("tsmean(df,'Test1')")
    showeval("df['Test2'].mean() # wrongas expected")
    showeval("df['Test2'].mean(weighted=df['dt']) # wrong")
    showeval("df['Test2'].mean(weighted=df['tTest2']) # wrong")
    showeval("(df['Test2']*df['tTest2']).sum()/df['tTest2'].sum() # ok")
    showeval("tsmean(df,'Test2')")

def rest2():
    df = makedt(getdf(['data/*SkyCam1*', 'data/*Fed3Pact*']))

def rest():
    '''Just a block to keep scrap code in'''
    # tidy up the data
    if False:
        df['SkyCam1_2mOk'] = df['SkyCam1_2mOk'].apply(
            lambda x: offset(limit(x, 0, 0.5), -1))
        df['SkyCam2_2mOk'] = df['SkyCam2_2mOk'].apply(
            lambda x: offset(limit(x, 0, 0.5), -2))
        df['SkyCam3_2mOk'] = df['SkyCam3_2mOk'].apply(
            lambda x: offset(limit(x, 0, 0.5), -3))

        df['Fed3Pact'] = df['Fed3Pact'].apply(lambda x: scale(x, 0.1))

        # # df['SkyCam2_2mOk'] = df['SkyCam2_2mOk'].apply(lambda x: limit(x,0,1))
        # # df['SkyCam3_2mOk'] = df['SkyCam3_2mOk'].apply(lambda x: limit(x,0,1))


        # do some basic statistics
        if False:
            for v in tsvars():
                print(tssummary(df, v))

                if False:
                    tb = 0
                    for (t, v) in hists['SkyCam2_2mOk']:
                        print((t - tb), v)
                        tb = t

                        # do some plots
                        # plt.ion() -- interactive

                        if False:
                            for v in tsvars():
                                df[v].plot(kind='line',
                                           drawstyle='steps',
                                           title='title',
                                           xlim=(pd.datetime(2015, 10, 18, 19),
                                                 pd.datetime(2015, 12, 25, 10)))
                                plt.show()

                                if False:
                                    plotPdf('daily.pdf')


                                # df.csv_export('dataexport.csv')


# finally call main (or profile it)
if __name__ == '__main__':
    if opts.get('-profile_main'):
        profile('main()')
    else:
        main()
