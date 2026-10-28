#pragma once
// Generated from ina_decision_tree.joblib. Do not edit manually.
#include <stddef.h>

static const int INA_MODEL_NUM_FEATURES = 12;
static const int INA_MODEL_NUM_CLASSES = 2;
static const char* const INA_MODEL_CLASS_NAMES[] = {
  "AIRFLOW_RESTRICTED",
  "NORMAL",
};

// Feature order expected by the trained model:
// x[0] = voltage_V_mean
// x[1] = voltage_V_std
// x[2] = voltage_V_min
// x[3] = voltage_V_max
// x[4] = current_mA_mean
// x[5] = current_mA_std
// x[6] = current_mA_min
// x[7] = current_mA_max
// x[8] = power_calc_mW_mean
// x[9] = power_calc_mW_std
// x[10] = power_calc_mW_min
// x[11] = power_calc_mW_max

inline int inaModelPredict(const float* x) {
  if (x[7] <= 92.4000015f) {
    return 0; // AIRFLOW_RESTRICTED
  } else {
    return 1; // NORMAL
  }
}
