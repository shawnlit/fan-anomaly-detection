#include <Wire.h>
#include <math.h>
#include <string.h>
#include <Adafruit_INA219.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "ina_model.h"

Adafruit_INA219 ina219;

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDR 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

const int BUZZER_PIN = 27;
const unsigned long SAMPLE_INTERVAL_MS = 100; // Approx. 10 Hz
const int SAMPLES_PER_WINDOW = 50;            // 50 samples ~= 5 seconds

unsigned long lastSampleMs = 0;
int n = 0;

float vSum = 0, vSqSum = 0, vMin = 1e9f, vMax = -1e9f;
float iSum = 0, iSqSum = 0, iMin = 1e9f, iMax = -1e9f;
float pSum = 0, pSqSum = 0, pMin = 1e9f, pMax = -1e9f;

char lastPrediction[32] = "WARMING UP";
unsigned long lastDisplayMs = 0;
const unsigned long DISPLAY_INTERVAL_MS = 200;

void resetWindow() {
  n = 0;

  vSum = 0;
  vSqSum = 0;
  vMin = 1e9f;
  vMax = -1e9f;

  iSum = 0;
  iSqSum = 0;
  iMin = 1e9f;
  iMax = -1e9f;

  pSum = 0;
  pSqSum = 0;
  pMin = 1e9f;
  pMax = -1e9f;
}

float safeStd(float sum, float sqSum, int count) {
  if (count <= 0) return 0;

  float mean = sum / count;
  float variance = (sqSum / count) - (mean * mean);

  // Avoid a tiny negative value caused by floating-point rounding.
  if (variance < 0) variance = 0;

  return sqrtf(variance);
}

void addValue(float v, float i, float p) {
  vSum += v;
  vSqSum += v * v;
  if (v < vMin) vMin = v;
  if (v > vMax) vMax = v;

  iSum += i;
  iSqSum += i * i;
  if (i < iMin) iMin = i;
  if (i > iMax) iMax = i;

  pSum += p;
  pSqSum += p * p;
  if (p < pMin) pMin = p;
  if (p > pMax) pMax = p;

  n++;
}

void updateOLED(float current_mA, const char* prediction) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("INA219 FAN MONITOR");

  display.drawLine(0, 11, 127, 11, SSD1306_WHITE);

  display.setCursor(0, 16);
  display.println("Current:");

  display.setTextSize(2);
  display.setCursor(0, 27);
  display.print(current_mA, 1);
  display.print(" mA");

  display.setTextSize(1);
  display.drawLine(0, 49, 127, 49, SSD1306_WHITE);
  display.setCursor(0, 54);
  display.print("ML: ");
  display.print(prediction);

  display.display();
}

void setup() {
  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Wire.begin(21, 22);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    Serial.println("ERROR: OLED not detected. Check wiring/address.");
    while (true) delay(1000);
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("Starting INA219 ML...");
  display.display();

  if (!ina219.begin()) {
    Serial.println("ERROR: INA219 not detected. Check wiring and I2C.");
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("INA219 ERROR");
    display.println("Check wiring");
    display.display();

    while (true) delay(1000);
  }

  // Keep calibration consistent with the data collection firmware.
  ina219.setCalibration_32V_2A();

  Serial.println("INA219 ML classifier with OLED started");
  Serial.println("Current refreshes continuously; ML prediction every 5 seconds.");
  Serial.println("window_ms,voltage_mean_V,current_mean_mA,current_max_mA,power_mean_mW,prediction");

  resetWindow();
  lastSampleMs = millis();
}

void loop() {
  unsigned long now = millis();

  if (now - lastSampleMs < SAMPLE_INTERVAL_MS) {
    delay(1);
    return;
  }
  lastSampleMs = now;

  float voltage_V = ina219.getBusVoltage_V();
  float current_mA = ina219.getCurrent_mA();

  if (!isfinite(voltage_V) || !isfinite(current_mA) || current_mA < -10.0f) {
    Serial.println("WARNING: invalid INA219 reading; sample skipped");
    return;
  }

  // V * mA = mW. The inconsistent INA219 power register is not used.
  float power_mW = voltage_V * current_mA;
  addValue(voltage_V, current_mA, power_mW);

  // Update OLED independently of the five-second classification window.
  if (now - lastDisplayMs >= DISPLAY_INTERVAL_MS) {
    updateOLED(current_mA, lastPrediction);
    lastDisplayMs = now;
  }

  if (n >= SAMPLES_PER_WINDOW) {
    float vMean = vSum / n;
    float vStd = safeStd(vSum, vSqSum, n);
    float iMean = iSum / n;
    float iStd = safeStd(iSum, iSqSum, n);
    float pMean = pSum / n;
    float pStd = safeStd(pSum, pSqSum, n);

    // Feature order must match the model's feature_columns:
    // voltage stats, current stats, calculated-power stats;
    // each group ordered mean, std, min, max.
    float features[12] = {
      vMean, vStd, vMin, vMax,
      iMean, iStd, iMin, iMax,
      pMean, pStd, pMin, pMax
    };

    int classIndex = inaModelPredict(features);
    const char* label = "UNKNOWN";

    if (classIndex >= 0 && classIndex < INA_MODEL_NUM_CLASSES) {
      label = INA_MODEL_CLASS_NAMES[classIndex];
    }

    strncpy(lastPrediction, label, sizeof(lastPrediction) - 1);
    lastPrediction[sizeof(lastPrediction) - 1] = '\0';

    Serial.print(millis());
    Serial.print(",");
    Serial.print(vMean, 4);
    Serial.print(",");
    Serial.print(iMean, 3);
    Serial.print(",");
    Serial.print(iMax, 3);
    Serial.print(",");
    Serial.print(pMean, 3);
    Serial.print(",");
    Serial.println(label);

    if (strcmp(label, "AIRFLOW_RESTRICTED") == 0) {
      tone(BUZZER_PIN, 2000, 250);
    } else {
      noTone(BUZZER_PIN);
      digitalWrite(BUZZER_PIN, LOW);
    }

    // Refresh OLED immediately so the new prediction appears promptly.
    updateOLED(current_mA, lastPrediction);
    lastDisplayMs = millis();

    resetWindow();
  }
}
