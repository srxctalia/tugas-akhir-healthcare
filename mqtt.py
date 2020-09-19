import paho.mqtt.client as mqtt
import csv
import sys
import pandas as pd
import time
import pandas as pd
import joblib
import pywt
import pywt.data
import numpy as np
import time
import matplotlib.pyplot as plt
import json

max=250

# Define event callbacks
def on_connect(client, userdata, flags, rc):
    # print("rc: " + str(rc))

    if rc == 0 :
        print("Connected to Server")
        global connected
        connected = True
    else :
        print("Connection failed")

# def on_data_full():
#     if len(data) == max:       #banyak sample
#         with open('Data Pengujian/Data3.csv', 'wb') as csvfile:
#             filewriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#             # filewriter.writerow(['adl'])
#             for i in data:
#                 filewriter.writerow([i])
#         sys.exit()

def on_message(client, obj, msg):
    print(msg.payload)

def on_publish(client, obj, mid):
    print("mid: " + str(mid))

def on_subscribe(client, obj, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

def on_log(client, obj, level, string):
    print(string)

connected = False

mqttc = mqtt.Client()
# Assign event callbacks
mqttc.on_message = on_message
mqttc.on_connect = on_connect

# Parse CLOUDMQTT_URL (or fallback to localhost)
#url_str = os.environ.get('CLOUDMQTT_URL', 'mqtt://localhost:1883')
#url = urlparse.urlparse(url_str)
topic = 'Topic'

# Connect
mqttc.connect("192.168.1.13", 1883  )
mqttc.on_message
# Start subscribe, with QoS level 0
mqttc.subscribe(topic, 0)

# Publish a message
#mqttc.publish(topic, i)

# Continue the network loop, exit when an error occurs
try :
    mqttc.loop_forever()
except KeyboardInterrupt  :
    mqttc.disconnect()


