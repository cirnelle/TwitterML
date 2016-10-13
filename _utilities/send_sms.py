#!/usr/local/bin/python3.4

import os
import sys
import subprocess
import twilio
from twilio.rest import TwilioRestClient
import time

#-------------------------------
# Twilio API, to send sms alerts
#-------------------------------

ACCOUNT_SID = 'AC817c734cbe1b13d2bbe97940e4efc413'
AUTH_TOKEN = 'd927e3830852ec14b06b302960833645'

twilio_client = TwilioRestClient(ACCOUNT_SID, AUTH_TOKEN)

#-------------------
# streaming db 1

try:

    subprocess.check_output("pgrep -f streaming_db_1.py", shell=True, universal_newlines=True)

except Exception as e:

    print()
    print('####################')

    now = time.strftime("%c")
    print("Time stamp: %s" % now)

    print(e)
    print("restarting streaming 1 ...")

    args = ['nohup', 'python3.4', '/Users/yi-linghwong/GitHub/TwitterML/_utilities/streaming_db_1.py', '2>&1', '&']

    with open("/Users/yi-linghwong/GitHub/TwitterML/_utilities/nohup.out",'a') as out:

        subprocess.Popen(args, stderr=out, stdout=out)

    if e.returncode == 1:
        print("exit code 1")

        # twilio_client.messages.create(
        #     to='+61406815706',
        #     from_='+61447752987',
        #     body='streaming db 1 stopped running'
        # )

#-------------------
# streaming db 2

try:

    subprocess.check_output("pgrep -f streaming_db_2.py", shell=True, universal_newlines=True)

except Exception as e:

    print()
    print('####################')

    now = time.strftime("%c")
    print("Time stamp: %s" % now)

    print(e)
    print ("restarting streaming 2 ...")

    args = ['nohup', 'python3.4', '/Users/yi-linghwong/GitHub/TwitterML/_utilities/streaming_db_2.py','2>&1', '&']

    with open("/Users/yi-linghwong/GitHub/TwitterML/_utilities/nohup.out", 'a') as out:
        subprocess.Popen(args, stderr=out, stdout=out)

    if e.returncode == 1:

        print ("exit code 1")

        # twilio_client.messages.create(
        #     to='+61406815706',
        #     from_='+61447752987',
        #     body='streaming db 2 stopped running')

#-------------------
# streaming db 3

try:

    subprocess.check_output("pgrep -f streaming_db_3.py", shell=True, universal_newlines=True)

except Exception as e:

    print()
    print('####################')

    now = time.strftime("%c")
    print("Time stamp: %s" % now)

    print(e)
    print ("restarting streaming 3 ...")

    args = ['nohup', 'python3.4', '/Users/yi-linghwong/GitHub/TwitterML/_utilities/streaming_db_3.py','2>&1', '&']

    with open("/Users/yi-linghwong/GitHub/TwitterML/_utilities/nohup.out", 'a') as out:
        subprocess.Popen(args, stderr=out, stdout=out)

    if e.returncode == 1:

        print ("exit code 1")

        # twilio_client.messages.create(
        #     to='+61406815706',
        #     from_='+61447752987',
        #     body='streaming db 3 stopped running')

#-------------------
# streaming db 4

try:

    subprocess.check_output("pgrep -f streaming_db_4.py", shell=True, universal_newlines=True)

except Exception as e:

    print()
    print('####################')

    now = time.strftime("%c")
    print("Time stamp: %s" % now)

    print(e)
    print ("restarting streaming 4 ...")

    args = ['nohup', 'python3.4', '/Users/yi-linghwong/GitHub/TwitterML/_utilities/streaming_db_4.py','2>&1', '&']

    with open("/Users/yi-linghwong/GitHub/TwitterML/_utilities/nohup.out", 'a') as out:
        subprocess.Popen(args, stderr=out, stdout=out)

    if e.returncode == 1:

        print ("exit code 1")

        # twilio_client.messages.create(
        #     to='+61406815706',
        #     from_='+61447752987',
        #     body='streaming db 4 stopped running')

#-------------------
# streaming db 5

try:

    subprocess.check_output("pgrep -f streaming_db_5.py", shell=True, universal_newlines=True)

except Exception as e:

    print()
    print('####################')

    now = time.strftime("%c")
    print("Time stamp: %s" % now)

    print(e)
    print ("restarting streaming 5 ...")

    args = ['nohup', 'python3.4', '/Users/yi-linghwong/GitHub/TwitterML/_utilities/streaming_db_5.py','2>&1', '&']

    with open("/Users/yi-linghwong/GitHub/TwitterML/_utilities/nohup.out", 'a') as out:
        subprocess.Popen(args, stderr=out, stdout=out)

    if e.returncode == 1:

        print ("exit code 1")

        # twilio_client.messages.create(
        #     to='+61406815706',
        #     from_='+61447752987',
        #     body='streaming db 5 stopped running'
        #   )


#-------------------
# streaming db 6

try:

    subprocess.check_output("pgrep -f streaming_db_6.py", shell=True, universal_newlines=True)

except Exception as e:

    print()
    print('####################')

    now = time.strftime("%c")
    print("Time stamp: %s" % now)

    print(e)
    print ("restarting streaming 5 ...")

    args = ['nohup', 'python3.4', '/Users/yi-linghwong/GitHub/TwitterML/_utilities/streaming_db_6.py','2>&1', '&']

    with open("/Users/yi-linghwong/GitHub/TwitterML/_utilities/nohup.out", 'a') as out:
        subprocess.Popen(args, stderr=out, stdout=out)

    if e.returncode == 1:

        print ("exit code 1")

        # twilio_client.messages.create(
        #     to='+61406815706',
        #     from_='+61447752987',
        #     body='streaming db 6 stopped running'
        #   )






