"""Módulo principal: entrena un clasificador y calcula métricas."""
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def cargar_datos(test_size=0.2, semilla=42):
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=test_size, random_state=semilla, stratify=y)


def entrenar(X_train, y_train, semilla=42):
    modelo = DecisionTreeClassifier(max_depth=3, random_state=semilla)
    modelo.fit(X_train, y_train)
    return modelo


def calcular_metricas(y_real, y_pred):
    return {
        "accuracy": round(accuracy_score(y_real, y_pred), 4),
        "precision": round(precision_score(y_real, y_pred, average="macro"), 4),
        "recall": round(recall_score(y_real, y_pred, average="macro"), 4),
        "f1": round(f1_score(y_real, y_pred, average="macro"), 4),
    }


def main():
    X_train, X_test, y_train, y_test = cargar_datos()
    modelo = entrenar(X_train, y_train)
    metricas = calcular_metricas(y_test, modelo.predict(X_test))
    with open("metrics.json", "w") as f:
        json.dump(metricas, f, indent=2)
    print("=== Resultados del modelo ===")
    for nombre, valor in metricas.items():
        print(f"{nombre:>10}: {valor}")
    return metricas


if __name__ == "__main__":
    main()
