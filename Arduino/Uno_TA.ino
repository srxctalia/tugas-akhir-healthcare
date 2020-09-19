#include <Wire.h>
#include <ArduinoJson.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <NTPClient.h>
#include <WiFiUdp.h>

const char* ssid = "Hunian_bawang";
const char* password = "baubawang";
const char* mqtt_server = "192.168.1.13";
const int MPU=0x68;  // I2C address of the MPU-6050 Kaki Kanan 

float AcX,AcY,AcZ,Tmp,GyX,GyY,GyZ,currentTime; //Kanan



int i;

WiFiClient espClient;
PubSubClient client(espClient);

WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP);

int val = 0;
char data[10000];

String formattedDate;
String timeStamp;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.beginTransmission(MPU);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0x00);     // set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);
    
  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
  delay(20);
  i = 1;

  timeClient.begin();
  timeClient.setTimeOffset(25200);
}

void json(){
  sendStream();
  String payload = "{\"AcX_1\": "+ String(AcX) 
  +",\"AcY_1\": "+ String(AcY)
  +",\"AcZ_1\": "+ String(AcZ)
  +",\"GyX_1\": "+ String(GyX)
  +",\"GyY_1\": "+ String(GyY)
  +",\"GyZ_1\": "+ String(GyZ)
  +",\"Time\": "+ String(timeStamp)+"}";
  payload.toCharArray(data, (payload.length() + 1));
  client.publish("Topic", data);
}

void setup_wifi() {

  delay(10);
  // We start by connecting to a WiFi network
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void callback(char* topic, byte* payload, unsigned int length) {
//  Serial.print("Message arrived [");
//  Serial.print(topic);
//  Serial.print(" ");
  for (int i = 0; i < length; i++) {
    Serial.print((char)payload[i]);
  }
  Serial.println();
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Attempt to connect
    if (client.connect("ESP8266Client")) {
      Serial.println("connected");
      // Once connected, publish an announcement...
      //json();
      //client.publish(, input);
      // ... and resubscribe
      client.subscribe("Topic");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(10);
    }
  }
}
void sendStream()
{
  Wire.beginTransmission(MPU);
  Wire.write(0x3B);  // starting with register 0x3B (ACCEL_XOUT_H)
  Wire.endTransmission(false);
  Wire.requestFrom(MPU,14,true);  // request a total of 14 registers
  AcX=(Wire.read()<<8|Wire.read())/16384.0;  // 0x3B (ACCEL_XOUT_H) & 0x3C (ACCEL_XOUT_L)    
  AcY=(Wire.read()<<8|Wire.read())/16384.0;  // 0x3D (ACCEL_YOUT_H) & 0x3E (ACCEL_YOUT_L)
  AcZ=(Wire.read()<<8|Wire.read())/16384.0;  // 0x3F (ACCEL_ZOUT_H) & 0x40 (ACCEL_ZOUT_L)
  
  Wire.beginTransmission(MPU);
  Wire.write(0x43);  // starting with register 0x3B (ACCEL_XOUT_H)
  Wire.endTransmission(false);
  Wire.requestFrom(MPU,14,true);
  GyX=(Wire.read()<<8|Wire.read())/131.0;  // 0x43 (GYRO_XOUT_H) & 0x44 (GYRO_XOUT_L)
  GyY=(Wire.read()<<8|Wire.read())/131.0;  // 0x45 (GYRO_YOUT_H) & 0x46 (GYRO_YOUT_L)
  GyZ=(Wire.read()<<8|Wire.read())/131.0;  // 0x47 (GYRO_ZOUT_H) & 0x48 (GYRO_ZOUT_L)

  GyX = GyX + 0.56; // GyroErrorX ~(-0.56)
  GyY = GyY - 2; // GyroErrorY ~(2)
  GyZ = GyZ + 0.79; // GyroErrorZ ~ (-0.8)

  currentTime = millis() / 1000;
  
  Serial.print(" AcX = "); Serial.print(AcX);
  Serial.print(" | AcY = "); Serial.print(AcY);
  Serial.print(" | AcZ = "); Serial.print(AcZ);
  //Serial.print(" | Tmp = "); Serial.print(Tmp/340.00+36.53);  //equation for temperature in degrees C from datasheet
  Serial.print(" | GyX = "); Serial.print(GyX);
  Serial.print(" | GyY = "); Serial.print(GyY);
  Serial.print(" | GyZ = "); Serial.println(GyZ);
  Serial.print(" | Time = "); Serial.println(timeStamp);
  delay(500);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
  json();
  while(!timeClient.update()) {
    timeClient.forceUpdate();
  }
  formattedDate = timeClient.getFormattedTime();
  int splitT = formattedDate.indexOf("T");
  timeStamp = formattedDate.substring(splitT+1, formattedDate.length()-1);
  Serial.print("Time: ");Serial.println(timeStamp);
}
