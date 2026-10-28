# ESP32 TinyML Fan Anomaly Detection

A small embedded ML project that uses an ESP32 and INA219 sensor to monitor a 12V DC fan and classify its operating condition as `NORMAL` or `AIRFLOW_RESTRICTED`.

The project uses a Decision Tree trained in Python and deployed on the ESP32 for on-device inference. The predicted condition is displayed on an SSD1306 OLED display, with an audible buzzer alert triggered when restricted airflow is detected.

## Hardware

- ESP32 Development Board
- INA219 high-side voltage and current sensor (I2C)
- SSD1306 128x64 I2C OLED display (0x3C)
- Active/Passive Buzzer (GPIO 27)
- 12V DC cooling fan
- LM2596 DC-DC buck converter
- *(Optional data collection sensors: MPU6050 accelerometer/gyroscope, DS18B20 temperature sensor; not required for current INA219 inference)*

### Pin Configuration

| Component | ESP32 Pin | Notes |
| :--- | :--- | :--- |
| **INA219 SDA** | GPIO 21 | Shared I2C Data bus |
| **INA219 SCL** | GPIO 22 | Shared I2C Clock bus |
| **SSD1306 SDA** | GPIO 21 | Shared I2C Data bus |
| **SSD1306 SCL** | GPIO 22 | Shared I2C Clock bus |
| **Buzzer** | GPIO 27 | Active high alert output |

## Machine Learning

The model is a supervised Decision Tree classifier trained using Python and scikit-learn.

Sensor readings are collected under normal and airflow-restricted conditions. The raw time series is partitioned into 5-second windows (sampled at ~10 Hz), and 12 statistical features are extracted:

- **Voltage (V):** mean, standard deviation, minimum, maximum
- **Current (mA):** mean, standard deviation, minimum, maximum
- **Calculated Power (mW):** mean, standard deviation, minimum, maximum *(voltage × current)*

The trained model is exported into an embedded C++ header (`ina_model.h`) implementing an `inaModelPredict()` function that runs directly on the ESP32 without any external dependencies or cloud connectivity.

## Project Structure

```text
.
├── export_tree_to_cpp.py           # Script to convert trained .joblib decision tree into C++ header (ina_model.h)
├── ina_decision_tree.joblib        # Trained model artifact used by export_tree_to_cpp.py
├── ina-model/                      # Data processing, model training, raw datasets, and evaluation report
│   ├── fan_raw_dataset_normal.csv              # Raw INA219 sensor logs for normal fan operation
│   ├── fan_raw_dataset_airflow_restricted.csv  # Raw INA219 sensor logs with airflow restricted
│   ├── ina_pipeline.py                         # Training pipeline (feature extraction, train/test split, model training)
│   ├── requirements_ina_pipeline.txt           # Python dependencies (pandas, numpy, scikit-learn, joblib)
│   ├── ina_windowed_dataset.csv                # Extracted 5-second windowed feature dataset
│   ├── ina_decision_tree.joblib                # Trained DecisionTreeClassifier model artifact
│   └── ina_model_report.txt                    # Classification report, confusion matrix, and tree rules
├── ina219_esp32_ml/                # ESP32 Arduino sketch folder
│   ├── ina219_esp32_ml.ino         # Arduino firmware (sampling, feature extraction, on-device inference, OLED, buzzer)
│   └── ina_model.h                 # Exported C++ model header with inaModelPredict()
└── README.md                       # Project documentation
```

## Setup & Workflow

### 1. Python Environment

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ina-model/requirements_ina_pipeline.txt
```

### 2. Model Training & Evaluation

Run the training pipeline to generate the windowed feature dataset, train the decision tree, and generate the evaluation report:

```bash
python ina-model/ina_pipeline.py
```

Outputs generated inside `ina-model/`:
- `ina-model/ina_windowed_dataset.csv`: 12-feature windowed dataset.
- `ina-model/ina_decision_tree.joblib`: Trained scikit-learn model.
- `ina-model/ina_model_report.txt`: Model accuracy, classification report, and decision rules.

### 3. Export Model to C++

To convert the trained model into a C++ header for the ESP32:

```bash
# Ensure ina_decision_tree.joblib is in the root directory (or copied from ina-model/)
cp ina-model/ina_decision_tree.joblib .
python export_tree_to_cpp.py
```

This generates `ina_model.h`. Copy it into the Arduino sketch directory:

```bash
cp ina_model.h ina219_esp32_ml/
```

### 4. ESP32 Firmware Deployment

1. Open `ina219_esp32_ml/ina219_esp32_ml.ino` in the Arduino IDE.
2. In the Arduino Library Manager (`Tools` -> `Manage Libraries...`), install:
   - **Adafruit INA219**
   - **Adafruit SSD1306**
   - **Adafruit GFX Library**
3. Select your ESP32 board (e.g., `ESP32 Dev Module`) and corresponding COM / serial port.
4. Upload the sketch to the ESP32.
5. Open the Serial Monitor at **115200 baud** to view live sampling stats and classification logs.

## Limitations

The model is trained for two experimental fan conditions. Its performance depends on the collected data and testing conditions. More independent recordings across varying fan models, voltages, and ambient operating conditions are recommended to further evaluate generalization.

## License

This project is for educational and experimental purposes.
