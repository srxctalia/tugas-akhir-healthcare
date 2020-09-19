import paho.mqtt.client as mqtt
import sys
import pandas as pd
import time
import joblib
import pywt
import pywt.data
import numpy as np
import matplotlib.pyplot as plt
import json
from realtimelib import classify


list_data = []
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

def on_message(client, obj, msg):
    global list_data
    data = (msg.payload)
    # print(data)
    fak = (data.decode("utf-8"))
    brm = "'"+fak+"'"
    acx, acy, acz, gyx, gyy, gyz, _ = brm.split(",")
    _,acx = acx.split(":")
    _,acy = acy.split(":")
    _,acz = acz.split(":")
    _,gyx = gyx.split(":")
    _,gyy = gyy.split(":")
    _,gyz = gyz.split(":")

    acx,acy,acz,gyx,gyy,gyz = float(acx), float(acy), float(acz), float(gyx), float(gyy), float(gyz)
    # print(type(acx))
    datas = [acx,acy,acz,gyx,gyy,gyz]
    list_data.append(datas)
    print(datas)
    if (len(list_data) > 25):
        predicted = classify(list_data)
        list_data = []
        print(predicted)

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
mqttc.connect("192.168.43.238", 1883  )
# mqttc.on_message
# Start subscribe, with QoS level 0
mqttc.subscribe(topic, 0)

# Publish a message
#mqttc.publish(topic, i)

# Continue the network loop, exit when an error occurs
try :
    mqttc.loop_forever()
except KeyboardInterrupt  :
    mqttc.disconnect()


