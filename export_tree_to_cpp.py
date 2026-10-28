#!/usr/bin/env python3
"""Export a scikit-learn DecisionTreeClassifier saved by ina_pipeline.py to C++."""
from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "ina_decision_tree.joblib"
HEADER_PATH = Path(__file__).resolve().parent / "ina_model.h"

def cpp_float(x):
    return f"{float(x):.9g}f"

def emit_node(tree, node_id, class_names, indent="  "):
    left = int(tree.children_left[node_id])
    right = int(tree.children_right[node_id])
    if left == right:  # leaf
        counts = np.asarray(tree.value[node_id]).reshape(-1)
        class_idx = int(np.argmax(counts))
        return f"{indent}return {class_idx}; // {class_names[class_idx]}\n"
    feat = int(tree.feature[node_id])
    threshold = cpp_float(tree.threshold[node_id])
    code = f"{indent}if (x[{feat}] <= {threshold}) {{\n"
    code += emit_node(tree, left, class_names, indent + "  ")
    code += f"{indent}}} else {{\n"
    code += emit_node(tree, right, class_names, indent + "  ")
    code += f"{indent}}}\n"
    return code

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}. Copy your trained ina_decision_tree.joblib "
            "into the same folder as this exporter."
        )
    payload = joblib.load(MODEL_PATH)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_names = payload.get("feature_columns", [])
    else:
        model = payload
        feature_names = []

    if not hasattr(model, "tree_") or not hasattr(model, "classes_"):
        raise TypeError("The loaded file is not a fitted scikit-learn DecisionTreeClassifier.")

    n_features = int(model.n_features_in_)
    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError("Feature-name count does not match model input count.")

    classes = [str(c) for c in model.classes_]
    tree = model.tree_

    lines = [
        "#pragma once",
        "// Generated from ina_decision_tree.joblib. Do not edit manually.",
        "#include <stddef.h>",
        "",
        f"static const int INA_MODEL_NUM_FEATURES = {n_features};",
        f"static const int INA_MODEL_NUM_CLASSES = {len(classes)};",
        "static const char* const INA_MODEL_CLASS_NAMES[] = {",
    ]
    for c in classes:
        escaped = c.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  "{escaped}",')
    lines += ["};", "", "// Feature order expected by the trained model:"]
    for i, name in enumerate(feature_names):
        lines.append(f"// x[{i}] = {name}")
    lines += [
        "",
        "inline int inaModelPredict(const float* x) {",
    ]
    lines.append(emit_node(tree, 0, classes, "  ").rstrip())
    lines += ["}", ""]
    HEADER_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Exported {HEADER_PATH.name}")
    print(f"Features: {n_features}")
    print("Feature order:")
    for i, name in enumerate(feature_names):
        print(f"  x[{i}] = {name}")
    print("Classes:")
    for i, c in enumerate(classes):
        print(f"  {i}: {c}")
    print(f"Tree nodes: {tree.node_count}")

if __name__ == "__main__":
    main()
