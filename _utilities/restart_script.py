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

    subprocess.check_output("pgrep -f stream_listen.py", shell=True, universal_newlines=True)

except Exception as e:

    print()
    print('####################')

    now = time.strftime("%c")
    print("Time stamp: %s" % now)

    print(e)
    print("restarting stream listen ...")

    args = ['nohup', 'python3.4', '/Users/yi-linghwong/GitHub/TwitterML/_utilities/stream_listen.py', '2>&1', '&']

    with open("/Users/yi-linghwong/GitHub/TwitterML/_utilities/nohup.out",'a') as out:

        subprocess.Popen(args, stderr=out, stdout=out)

    if e.returncode == 1:
        print("exit code 1")

        # twilio_client.messages.create(
        #     to='+61406815706',
        #     from_='+61447752987',
        #     body='streaming db 1 stopped running'
        # )

