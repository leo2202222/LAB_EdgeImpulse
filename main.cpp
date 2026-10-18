#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <MPU6050_light.h>
#include <Thingy_53_-_Predictive_Maintenance_inferencing.h>

// --- Configuration ---
char WIFI_SSID[] = "Wokwi-GUEST";
char WIFI_PASSWORD[] = "";      
const char* token = "3gH11dDfSJNcOx0McMdk"; // <-- Replace with your Access Token
const char* thingsboard_server = "thingsboard.cloud"; // <-- Replace with your ThingsBoard server address
// if DNS fail use demo.thingsboard.io[104.196.24.70] | thingsboard.cloud[44.194.165.190] | eu.thingsboard.cloud[3.69.110.78]
// or try ping <domain name> to get IP address from CMD

const int port = 1883;

const char STUDENT_ID[]      = "68096166";   // <-- Replace with your Student ID
const char FIRMWARE_VERSION[] = "1.0"; 

// --- Hardware Pins (from diagram.json — ESP32-S3) ---
#define pinLED 4       // LED cathode → GPIO 4 (active-LOW: sink to turn ON)
#define LED_ON  LOW    // LED wired 3V3 → R → Anode → Cathode → GPIO: LOW = ON
#define LED_OFF HIGH

bool ledState = false;  // current LED state (true = ON), reported as "led_state" attribute

#define pinSDA 8       // MPU6050 SDA → GPIO 8 (10kΩ pull-up to 3V3)
#define pinSCL 9       // MPU6050 SCL → GPIO 9 (10kΩ pull-up to 3V3)

// --- MPU6050 Configuration ---
#define NUM_MPUS 1
MPU6050 mpu[NUM_MPUS] = { MPU6050(Wire) };
const int AD0pin[NUM_MPUS] = {5};   // AD0 → GPIO 5

// --- Peripheral Objects ---
WiFiClient espClient;
PubSubClient client(espClient);

// --- Timing ---
unsigned long lastMPUPrintTime = 0;
const unsigned long MPUPrintInterval = 1000;

unsigned long lastMPUPublishTime = 0;
const unsigned long MPUPublishInterval = 1000;

// --- WiFi / MQTT non-blocking reconnect state ---
unsigned long lastWiFiRetry = 0;
const unsigned long wifiRetryInterval = 500;

unsigned long lastMQTTRetry = 0;
const unsigned long mqttRetryInterval = 5000;

// --- Edge Impulse inference ---
// MPU6050_light reports acceleration in g, while the deployed model expects m/s^2.
const float GRAVITY_MS2 = 9.80665f;
const float MAX_ACCELERATION_G = 2.0f; // Edge Impulse examples limit input to +/-2g
const float UNBALANCE_THRESHOLD = EI_CLASSIFIER_THRESHOLD; // 0.60 from the model
const float NORMAL_THRESHOLD = EI_CLASSIFIER_THRESHOLD;    // 0.60 from the model
const float ANOMALY_SCORE_THRESHOLD = 0.0f;
const uint32_t INFERENCE_SAMPLE_INTERVAL_US =
    (uint32_t)(EI_CLASSIFIER_INTERVAL_MS * 1000.0f);

float inferenceBuffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
size_t inferenceBufferIndex = 0;
uint32_t nextInferenceSampleUs = 0;

// Keep the latest anomaly until ThingsBoard confirms that MQTT accepted it.
bool pendingAnomalyTelemetry = false;
char pendingFault[24] = "";
float pendingAnomalyScore = 0.0f;
float pendingUnbalanceProbability = 0.0f;
float pendingAccX = 0.0f;
float pendingAccY = 0.0f;
float pendingAccZ = 0.0f;

// --- Prototypes ---
void onMessage(char* topic, byte* payload, unsigned int length);
void connectToWiFi();
void connectToMQTTBroker();
void setupMPU();
void updateMPU();
void readMPU();
void SelectMPU(int selection);
void publishMPUTelemetry();
void publishAttributes();
void collectInferenceSample();
void runMachineInference();
bool publishAnomalyTelemetry(const char* fault, float anomalyScore,
                             float unbalanceProbability, float accX,
                             float accY, float accZ);
void sendPendingAnomalyTelemetry();

// ====================================================================
//  setup()
// ====================================================================
void setup() {
  Serial.begin(115200);  // ESP32-S3 convention; matches boot ROM baud rate
  pinMode(pinLED, OUTPUT);
  digitalWrite(pinLED, LED_OFF);

  // --- Initialize I2C and MPU6050 ---
  setupMPU();

  run_classifier_init();
  nextInferenceSampleUs = micros();
  Serial.printf("Edge Impulse ready: %u samples, %.1f ms interval\n",
                EI_CLASSIFIER_RAW_SAMPLE_COUNT,
                (float)EI_CLASSIFIER_INTERVAL_MS);

  // --- WiFi (non-blocking initial connect) ---
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  // Allow up to ~10s for initial connect, then fall through to loop()
  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 10000) {
    delay(250);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ Connected to WiFi");
  } else {
    Serial.println("\n⚠️ WiFi not connected yet — will retry in loop()");
  }

  client.setServer(thingsboard_server, port);
  client.setBufferSize(512);  // headroom for 10-field MPU JSON + MQTT framing
  // *** set Callback receive from Server ***
  client.setCallback(onMessage);
}

// ====================================================================
//  loop()
// ====================================================================
void loop() {
  // --- Original non-blocking WiFi reconnect flow ---
  if (WiFi.status() != WL_CONNECTED) {
    connectToWiFi();
    return;  // wait for WiFi before MQTT, sensing and inference
  }

  // --- Original non-blocking MQTT reconnect flow ---
  if (!client.connected()) {
    connectToMQTTBroker();
    return;  // wait for ThingsBoard before sensing and inference
  }
  client.loop();

  // Retry a previously failed anomaly publish after MQTT maintenance.
  sendPendingAnomalyTelemetry();

  // Keep MPU complementary filter running every iteration
  updateMPU();

  // Assignment addition: collect samples and run Edge Impulse inference.
  collectInferenceSample();

  // Serial debug output (gated by MPUPrintInterval)
  readMPU();

  /*
   * ORIGINAL MPU TELEMETRY (kept for reference)
   *
   * The assignment requires no telemetry while the machine is normal and
   * explicitly asks to stop the old raw-MPU telemetry sent every second.
   * publishMPUTelemetry();
   */
}

// ====================================================================
//  Edge Impulse Predictive-maintenance Inference
// ====================================================================

// Collect one XYZ accelerometer frame at the model's required 100 Hz rate.
// The signed subtraction makes micros() rollover safe.
void collectInferenceSample() {
  uint32_t now = micros();
  if ((int32_t)(now - nextInferenceSampleUs) < 0) return;

  nextInferenceSampleUs = now + INFERENCE_SAMPLE_INTERVAL_US;

  if (inferenceBufferIndex + EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME >
      EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
    inferenceBufferIndex = 0;
  }

  /*
   * ORIGINAL INPUT (kept for reference)
   * inferenceBuffer[inferenceBufferIndex++] = mpu[0].getAccX() * GRAVITY_MS2;
   * inferenceBuffer[inferenceBufferIndex++] = mpu[0].getAccY() * GRAVITY_MS2;
   * inferenceBuffer[inferenceBufferIndex++] = mpu[0].getAccZ() * GRAVITY_MS2;
   */

  // Match the Edge Impulse accelerometer examples by clipping each axis to
  // the model's accepted +/-2g range before converting g to m/s^2.
  float accX = constrain(mpu[0].getAccX(), -MAX_ACCELERATION_G, MAX_ACCELERATION_G);
  float accY = constrain(mpu[0].getAccY(), -MAX_ACCELERATION_G, MAX_ACCELERATION_G);
  float accZ = constrain(mpu[0].getAccZ(), -MAX_ACCELERATION_G, MAX_ACCELERATION_G);
  inferenceBuffer[inferenceBufferIndex++] = accX * GRAVITY_MS2;
  inferenceBuffer[inferenceBufferIndex++] = accY * GRAVITY_MS2;
  inferenceBuffer[inferenceBufferIndex++] = accZ * GRAVITY_MS2;

  if (inferenceBufferIndex >= EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
    runMachineInference();
    inferenceBufferIndex = 0;
    // Do not try to catch up samples missed while the classifier was running.
    nextInferenceSampleUs = micros() + INFERENCE_SAMPLE_INTERVAL_US;
  }
}

void runMachineInference() {
  signal_t signal;
  int signalError = numpy::signal_from_buffer(
      inferenceBuffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
  if (signalError != 0) {
    Serial.printf("Inference signal error: %d\n", signalError);
    return;
  }

  ei_impulse_result_t result = {0};
  EI_IMPULSE_ERROR inferenceError = run_classifier(&signal, &result, false);
  if (inferenceError != EI_IMPULSE_OK) {
    Serial.printf("Inference failed: %d\n", inferenceError);
    return;
  }

  float normalProbability = 0.0f;
  float unbalanceProbability = 0.0f;
  for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    const char* label = result.classification[i].label;
    if (strcmp(label, "Normal_Operation") == 0) {
      normalProbability = result.classification[i].value;
    } else if (strcmp(label, "Unbalance_In_Rotating_Parts") == 0) {
      unbalanceProbability = result.classification[i].value;
    }
  }

  float anomalyScore = result.anomaly;
  const char* machineState = "normal";
  const char* fault = nullptr;

  /*
   * ORIGINAL ANOMALY-FIRST DECISION (kept for reference)
   *
   * if (unbalanceProbability >= UNBALANCE_THRESHOLD) {
   *   machineState = "anomaly";
   *   fault = "unbalance";
   * } else if (anomalyScore > ANOMALY_SCORE_THRESHOLD) {
   *   machineState = "anomaly";
   *   fault = "unknown_anomaly";
   * }
   */

  // Wokwi decision: trust a confident known class first. This prevents a
  // stationary simulated MPU from becoming unknown_anomaly on every result.
  if (unbalanceProbability >= UNBALANCE_THRESHOLD) {
    machineState = "anomaly";
    fault = "unbalance";
  } else if (normalProbability >= NORMAL_THRESHOLD) {
    machineState = "normal";
    fault = nullptr;
  } else if (anomalyScore > ANOMALY_SCORE_THRESHOLD) {
    machineState = "anomaly";
    fault = "unknown_anomaly";
  }

  bool isAnomaly = (fault != nullptr);
  bool ledChanged = (ledState != isAnomaly);
  ledState = isAnomaly;
  digitalWrite(pinLED, ledState ? LED_ON : LED_OFF);

  // Required serial status after every completed inference.
  Serial.println();
  Serial.println(F("==================== AI INFERENCE RESULT ===================="));
  if (isAnomaly) {
    Serial.printf("🔴 Machine state: anomaly | fault=%s | normal=%.3f "
                  "unbalance=%.3f anomaly_score=%.3f\n",
                  fault, normalProbability, unbalanceProbability, anomalyScore);
  } else {
    Serial.printf("🟢 Machine state: normal | normal=%.3f unbalance=%.3f "
                  "anomaly_score=%.3f\n",
                  normalProbability, unbalanceProbability, anomalyScore);
  }
  Serial.println(F("============================================================="));
  Serial.println();

  // Queue alert telemetry only for abnormal inference results. It is not
  // discarded if WiFi/MQTT happens to be offline during this inference.
  if (isAnomaly) {
    strncpy(pendingFault, fault, sizeof(pendingFault) - 1);
    pendingFault[sizeof(pendingFault) - 1] = '\0';
    pendingAnomalyScore = anomalyScore;
    pendingUnbalanceProbability = unbalanceProbability;
    pendingAccX = mpu[0].getAccX();
    pendingAccY = mpu[0].getAccY();
    pendingAccZ = mpu[0].getAccZ();
    pendingAnomalyTelemetry = true;

    // Try immediately so the telemetry status belongs to this inference
    // result in Serial. Keep it queued when MQTT is offline or publish fails.
    if (client.connected() &&
        publishAnomalyTelemetry(pendingFault, pendingAnomalyScore,
                                pendingUnbalanceProbability, pendingAccX,
                                pendingAccY, pendingAccZ)) {
      pendingAnomalyTelemetry = false;
      Serial.println(F("Telemetry status: SENT"));
    } else {
      Serial.println(client.connected()
                         ? F("Telemetry status: PUBLISH FAILED - WILL RETRY")
                         : F("Telemetry status: WAITING FOR MQTT"));
    }
  } else {
    Serial.println(F("Telemetry status: NOT SENT (machine is normal)"));
  }

  if (ledChanged && client.connected()) {
    publishAttributes();
  }
}

void sendPendingAnomalyTelemetry() {
  if (!pendingAnomalyTelemetry || !client.connected()) return;

  if (publishAnomalyTelemetry(pendingFault, pendingAnomalyScore,
                              pendingUnbalanceProbability, pendingAccX,
                              pendingAccY, pendingAccZ)) {
    pendingAnomalyTelemetry = false;
  }
}

bool publishAnomalyTelemetry(const char* fault, float anomalyScore,
                             float unbalanceProbability, float accX,
                             float accY, float accZ) {
  JsonDocument doc;
  doc["machine_state"] = "anomaly";
  doc["fault"] = fault;
  doc["anomaly_score"] = anomalyScore;
  doc["unbalance_prob"] = unbalanceProbability;
  doc["acc_x"] = accX;
  doc["acc_y"] = accY;
  doc["acc_z"] = accZ;

  char payload[256];
  size_t len = serializeJson(doc, payload, sizeof(payload));
  if (len > 0 && client.publish("v1/devices/me/telemetry", payload)) {
    Serial.print(F("Anomaly telemetry sent: "));
    Serial.println(payload);
    return true;
  } else {
    Serial.println(F("Anomaly telemetry publish failed"));
    return false;
  }
}

// ====================================================================
//  MPU6050 Functions
// ====================================================================

// Initialize I2C on diagram-specified pins and set up all MPU6050 sensors.
void setupMPU() {
  Wire.begin(pinSDA, pinSCL);
  Wire.setTimeOut(1000);  // guard against a wedged I2C bus (avoids interrupt-WDT panic)

  for (int i = 0; i < NUM_MPUS; i++) {
    pinMode(AD0pin[i], OUTPUT);
    digitalWrite(AD0pin[i], HIGH);  // default HIGH → address 0x69 (deselected)
  }

  Serial.println(F("Initializing MPU-6050 sensors. Do not move the sensor."));
  delay(500);

  for (int i = 0; i < NUM_MPUS; i++) {
    SelectMPU(i);

    byte status = mpu[i].begin();
    Serial.print(F("Initializing MPU-6050 "));
    Serial.print(i);
    Serial.print(F(", error = "));
    Serial.print(status);
    if (status == 0)
      Serial.print(F(" (no error)"));
    Serial.println();

    // calcOffsets() runs a tight 500-iteration I2C polling loop that hangs
    // Wokwi's simulated I2C bus and trips the interrupt watchdog — skip it
    // here; offsets default to 0 (set in the MPU6050 constructor).
  }

  Serial.println(F("MPU-6050 initialization done"));
  Serial.println(F("Click on a MPU-6050 module and change the acceleration and gyro."));

  lastMPUPrintTime = millis();
  lastMPUPublishTime = millis();
}

// Call every loop() iteration to keep the complementary filter updated.
void updateMPU() {
  for (int i = 0; i < NUM_MPUS; i++) {
    SelectMPU(i);
    mpu[i].update();
  }
}

// Print detailed sensor readings to Serial (non-blocking, interval-gated).
void readMPU() {
  if (millis() - lastMPUPrintTime < MPUPrintInterval) return;
  lastMPUPrintTime = millis();

  for (int i = 0; i < NUM_MPUS; i++) {
    SelectMPU(i);

    Serial.print(i);
    Serial.print(":");
    Serial.print(F("T="));
    Serial.print(mpu[i].getTemp(), 0);
    Serial.print(F(",ACC="));
    Serial.print(mpu[i].getAccX());
    Serial.print(",");
    Serial.print(mpu[i].getAccY());
    Serial.print(",");
    Serial.print(mpu[i].getAccZ());

    Serial.print(F(",GYRO="));
    Serial.print(mpu[i].getGyroX(), 0);
    Serial.print(",");
    Serial.print(mpu[i].getGyroY(), 0);
    Serial.print(",");
    Serial.print(mpu[i].getGyroZ(), 0);

    Serial.print(F(",ACC ANGLE="));
    Serial.print(mpu[i].getAccAngleX(), 0);
    Serial.print(",");
    Serial.print(mpu[i].getAccAngleY(), 0);

    Serial.print(F(",ANGLE="));
    Serial.print(mpu[i].getAngleX(), 0);
    Serial.print(",");
    Serial.print(mpu[i].getAngleY(), 0);
    Serial.print(",");
    Serial.print(mpu[i].getAngleZ(), 0);
    Serial.println();
  }
  Serial.println(F("======================================================================================\n"));
}

// Choose a MPU by index — drives AD0 LOW on the selected sensor (address 0x68),
// HIGH on all others (address 0x69).
void SelectMPU(int selection) {
  for (int i = 0; i < NUM_MPUS; i++) {
    if (i == selection)
      digitalWrite(AD0pin[i], LOW);   // selected → 0x68
    else
      digitalWrite(AD0pin[i], HIGH);  // not selected → 0x69
  }
}

// ====================================================================
//  ThingsBoard Telemetry — MPU6050
// ====================================================================

// Publish MPU6050 telemetry to ThingsBoard (independent timer from serial debug).
void publishMPUTelemetry() {
  if (millis() - lastMPUPublishTime < MPUPublishInterval) return;
  lastMPUPublishTime = millis();

  for (int i = 0; i < NUM_MPUS; i++) {
    SelectMPU(i);

    JsonDocument doc;
    doc["mpu_temp"]    = mpu[i].getTemp();
    doc["acc_x"]       = mpu[i].getAccX();
    doc["acc_y"]       = mpu[i].getAccY();
    doc["acc_z"]       = mpu[i].getAccZ();
    doc["gyro_x"]      = mpu[i].getGyroX();
    doc["gyro_y"]      = mpu[i].getGyroY();
    doc["gyro_z"]      = mpu[i].getGyroZ();
    doc["angle_x"]     = mpu[i].getAngleX();
    doc["angle_y"]     = mpu[i].getAngleY();
    doc["angle_z"]     = mpu[i].getAngleZ();

    char payload[256];
    size_t len = serializeJson(doc, payload, sizeof(payload));
    if (len > 0) {
      client.publish("v1/devices/me/telemetry", payload);
      Serial.print(F("📨 MPU Sent: "));
      Serial.println(payload);
    }
  }
}

// ====================================================================
//  RPC Callback Function
// ====================================================================
// this function active when Postman/ThingsBoard send API
void onMessage(char* topic, byte* payload, unsigned int length) {
  Serial.print(F("📩 Message arrived ["));
  Serial.print(topic);
  Serial.println("]");

  // Build response topic by replacing "request" with "response".
  // Copy it first: `topic` points into PubSubClient's shared buffer,
  // which any client.publish() call overwrites.
  char responseTopic[128];
  strncpy(responseTopic, topic, sizeof(responseTopic) - 1);
  responseTopic[sizeof(responseTopic) - 1] = '\0';
  char* reqPtr = strstr(responseTopic, "request");
  if (reqPtr) {
    // In-place replace: "request" (7 chars) → "response" (8 chars)
    // Shift tail right by 1 to make room
    size_t tailLen = strlen(reqPtr + 7);
    memmove(reqPtr + 8, reqPtr + 7, tailLen + 1);
    memcpy(reqPtr, "response", 8);
  }

  JsonDocument doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) {
    Serial.print(F("❌ JSON parse error: "));
    Serial.println(err.c_str());
    return;
  }

  const char* methodName = doc["method"];  // "setLed"
  bool params = doc["params"];             // true or false

  bool handled = false;
  if (methodName && strcmp(methodName, "setLed") == 0) {
    handled = true;
    ledState = params;
    digitalWrite(pinLED, ledState ? LED_ON : LED_OFF);
    Serial.print(F("💡 LED Status: "));
    Serial.println(ledState ? "ON" : "OFF");
  }

  // send Response back (Two-way RPC)
  JsonDocument resp;
  resp["success"] = handled;
  if (handled) resp["led_state"] = ledState;
  char respPayload[64];
  serializeJson(resp, respPayload, sizeof(respPayload));
  client.publish(responseTopic, respPayload);

  // Publish attributes only after the response — it reuses the MQTT buffer
  if (handled) publishAttributes();  // report new LED state to ThingsBoard
}

// ====================================================================
//  Non-blocking WiFi Reconnect
// ====================================================================
void connectToWiFi() {
  if (millis() - lastWiFiRetry < wifiRetryInterval) return;
  lastWiFiRetry = millis();

  Serial.print(".");
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ Connected to WiFi");
  }
}

// ====================================================================
//  Non-blocking MQTT Connection
// ====================================================================
void connectToMQTTBroker() {
  if (millis() - lastMQTTRetry < mqttRetryInterval) return;
  lastMQTTRetry = millis();

  Serial.println(F("☁️ Connecting to ThingsBoard..."));
  if (client.connect("ESP32_Client", token, NULL)) {
    Serial.println(F("✅ Connected"));
    // *** Subscribe topic receive RPC ***
    client.subscribe("v1/devices/me/rpc/request/+");
    publishAttributes();
  } else {
    Serial.print(F("❌ MQTT connect failed, rc="));
    Serial.println(client.state());
  }
}

// ====================================================================
//  ThingsBoard Attributes
// ====================================================================
void publishAttributes() {
  JsonDocument doc;
  doc["student_id"] = STUDENT_ID;
  doc["firmware_version"] = FIRMWARE_VERSION;
  doc["led_state"] = ledState;

  char payload[128];
  size_t len = serializeJson(doc, payload, sizeof(payload));

  if (len > 0 && client.publish("v1/devices/me/attributes", payload)) {
    Serial.print(F("📨 Attributes sent: "));
    Serial.println(payload);
  } else {
    Serial.println(F("❌ Attributes publish failed"));
  }
}
