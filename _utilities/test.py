#!/usr/bin/python3.4

import os
import sys
import psutil
import subprocess
import time

try:

    mongod_pid = subprocess.check_output(['pgrep', '-f', 'mongod'], universal_newlines=True)

    print (mongod_pid)

    process = psutil.Process(int(mongod_pid))

    mongod_memory = process.memory_info().rss

    print (mongod_memory)

    if mongod_memory > 16000000:

        print ("memory limit exceeded")
        process.terminate()

        time.sleep(5)

        print()
        print('####################')

        now = time.strftime("%c")
        print("Time stamp: %s" % now)

        print("restarting mongod...")

        FNULL = open(os.devnull, 'w')

        args = ['mongod', '--dbpath', os.path.expanduser('/data/db')]

        with open("/Users/yi-linghwong/GitHub/TwitterML/_utilities/nohup_mongod.out", 'a') as out:

            subprocess.Popen(args, stderr=out, stdout=out)

        print ("mongod restarted")


except Exception as e:

    print (e)

    print("mongod not running, starting mongod...")

    FNULL = open(os.devnull, 'w')

    args = ['mongod', '--dbpath', os.path.expanduser('/data/db')]

    with open("/Users/yi-linghwong/GitHub/TwitterML/_utilities/nohup_mongod.out", 'a') as out:

        subprocess.Popen(args, stderr=out, stdout=out)






