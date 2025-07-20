#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <SoftwareSerial.h>

// WiFi Credentials
const char* ssid = "agrosense";
const char* password = "";

// Raspberry Pi Flask Server Endpoint
const char* serverUrl = "http://10.42.0.1:5000/data";

// RS485 Communication Pins
#define RE D1
#define DE D2
SoftwareSerial mod(D5, D6);  // RS485 Serial (RX, TX)

// Modbus request frames
const byte soilSensorRequest[] = {0x01, 0x03, 0x00, 0x00, 0x00, 0x02, 0xC4, 0x0B};  
const byte phSensorRequest[] = {0x01, 0x03, 0x00, 0x00, 0x00, 0x01, 0x84, 0x0A};   
const byte ecRequest[] = {0x01, 0x03, 0x00, 0x15, 0x00, 0x01, 0x95, 0xCE};         
const byte salinityRequest[] = {0x01, 0x03, 0x00, 0x14, 0x00, 0x01, 0xC4, 0x0E};    

// Response buffer
byte responseBuffer[9];

// Sleep duration in microseconds (1 minute = 60000000 μs * 60 mins)
const uint64_t SLEEP_TIME = 3600000000;

void setup() {
    Serial.begin(4800);
    delay(1000);  // Give some time for serial to initialize
    
    mod.begin(4800);
    pinMode(RE, OUTPUT);
    pinMode(DE, OUTPUT);
    digitalWrite(RE, LOW);
    digitalWrite(DE, LOW);

    Serial.println("\n============================");
    Serial.println("  Soil Sensor Data Monitor  ");
    Serial.println("============================\n");
    Serial.println("Waking up, taking measurements...");

    // Connect to WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi...");
    int wifiAttempts = 0;
    while (WiFi.status() != WL_CONNECTED && wifiAttempts < 20) {
        delay(500);
        Serial.print(".");
        wifiAttempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi Connected!");
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
        
        // Collect and send sensor data
        collectAndSendData();
    } else {
        Serial.println("\nWiFi connection failed!");
    }

    // Prepare for deep sleep
    Serial.println("Going to deep sleep for 1 minute...");
    Serial.flush();
    delay(1000); 
    
    // Alternative approach using light sleep
    WiFi.disconnect(true);
    WiFi.mode(WIFI_OFF);
    WiFi.forceSleepBegin();
    delay(60000); // Sleep for 1 minute
    WiFi.forceSleepWake();
    ESP.reset(); // Reset after sleep
}

void loop() {
    // Nothing happens here with deep sleep approach
    ESP.reset();
}

// Collect and send sensor data
void collectAndSendData() {
    float moisture, temperature, ph;
    int ec, salinity;

    if (measureSoilMoistureAndTemperature(moisture, temperature)) {
        Serial.printf("Moisture: %.1f %%RH, Temperature: %.1f °C\n", moisture, temperature);
    } else {
        Serial.println("Failed to read moisture and temperature");
        moisture = -1;
        temperature = -1;
    }

    if (measureSoilPH(ph)) {
        Serial.printf("pH Level: %.2f\n", ph);
    } else {
        Serial.println("Failed to read pH");
        ph = -1;
    }

    if (measureSoilEC(ec)) {
        Serial.printf("Electrical Conductivity: %d µS/cm\n", ec);
    } else {
        Serial.println("Failed to read EC");
        ec = -1;
    }

    if (measureSoilSalinity(salinity)) {
        Serial.printf("Salinity: %d mg/L\n", salinity);
    } else {
        Serial.println("Failed to read salinity");
        salinity = -1;
    }

    sendDataToServer(moisture, temperature, ph, ec, salinity);
}

bool measureSoilMoistureAndTemperature(float &moisture, float &temperature) {
    if (sendModbusRequest(soilSensorRequest, 8, 7)) {
        moisture = ((responseBuffer[3] << 8) | responseBuffer[4]) / 10.0;
        int tempInt = (responseBuffer[5] << 8) | responseBuffer[6];
        if (tempInt >= 0x8000) tempInt -= 0x10000;
        temperature = tempInt / 10.0;
        return true;
    }
    return false;
}

bool measureSoilPH(float &ph) {
    if (sendModbusRequest(phSensorRequest, 8, 7)) {
        ph = ((responseBuffer[3] << 8) | responseBuffer[4]) / 100.0;
        return true;
    }
    return false;
}

bool measureSoilEC(int &ec) {
    if (sendModbusRequest(ecRequest, 8, 7)) {
        ec = (responseBuffer[3] << 8) | responseBuffer[4];
        return true;
    }
    return false;
}

bool measureSoilSalinity(int &salinity) {
    if (sendModbusRequest(salinityRequest, 8, 7)) {
        salinity = responseBuffer[4];
        return true;
    }
    return false;
}

bool sendModbusRequest(const byte *request, size_t requestSize, size_t expectedResponseSize) {
    for (int attempt = 0; attempt < 3; attempt++) {
        memset(responseBuffer, 0, sizeof(responseBuffer));
        mod.flush();

        digitalWrite(DE, HIGH);
        digitalWrite(RE, HIGH);
        delayMicroseconds(500);

        mod.write(request, requestSize);
        mod.flush();
        digitalWrite(DE, LOW);
        digitalWrite(RE, LOW);

        unsigned long startTime = millis();
        while (mod.available() < expectedResponseSize && millis() - startTime < 3000) {
            delay(10);
        }

        int receivedBytes = mod.available();
        if (receivedBytes >= expectedResponseSize) {
            for (size_t i = 0; i < expectedResponseSize; i++) {
                responseBuffer[i] = mod.read();
            }
            return true;
        }
        
        delay(200); // Small delay before retry
    }
    return false;
}

void sendDataToServer(float moisture, float temperature, float ph, int ec, int salinity) {
    if (WiFi.status() == WL_CONNECTED) {
        WiFiClient client;
        HTTPClient http;

        Serial.println("Sending data to server...");
        http.begin(client, serverUrl);
        http.addHeader("Content-Type", "application/json");

        String jsonData = "{";
        jsonData += "\"moisture\": " + String(moisture) + ", ";
        jsonData += "\"temperature\": " + String(temperature) + ", ";
        jsonData += "\"ph\": " + String(ph) + ", ";
        jsonData += "\"ec\": " + String(ec) + ", ";
        jsonData += "\"salinity\": " + String(salinity) + "}";

        Serial.println("JSON Data: " + jsonData);

        int httpResponseCode = http.POST(jsonData);
        Serial.print("HTTP Response code: ");
        Serial.println(httpResponseCode);

        http.end();
    } else {
        Serial.println("WiFi Disconnected! Data not sent.");
    }
}