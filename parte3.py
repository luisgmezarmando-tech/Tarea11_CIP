import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. CARGA DEL DATASET NUTT
# ---------------------------------------------------------
def cargar_nutt(ruta):
    """Carga Nutt (1).csv sin encabezados (28 patrones x 1070 atributos + clase)."""
    df = pd.read_csv(ruta, header=None)
    X = df.iloc[:, :1070].values.astype(float)
    y = df.iloc[:, 1070].values
    nombres = [f"pp{i + 1}" for i in range(len(df))]
    return X, y, nombres


# ---------------------------------------------------------
# 2. ENTRENAMIENTO (CÁLCULO DE CENTROIDES)
# ---------------------------------------------------------
def entrenar_euclidiano(X_train, y_train):
    """Calcula el centroide (promedio) de cada clase."""
    centroides = {}
    for clase in np.unique(y_train):
        filas_clase = X_train[y_train == clase]
        centroides[clase] = filas_clase.mean(axis=0)
    return centroides


# ---------------------------------------------------------
# 3. VALIDACIÓN LEAVE-ONE-OUT Y MÉTRICAS
# ---------------------------------------------------------
def evaluar_loo_euclidiano(X, y, nombres):
    filas_reporte = []
    TP = TN = FP = FN = 0
    n = len(X)

    for i in range(n):
        # Excluir patrón i para prueba; usar los 27 restantes para entrenamiento
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        x_test = X[i]
        clase_real = y[i]

        # Entrenar centroides con los 27 patrones y calcular distancias
        centroides = entrenar_euclidiano(X_train, y_train)
        distancias = {
            clase: np.linalg.norm(x_test - mu)
            for clase, mu in centroides.items()
        }

        prediccion = min(distancias, key=distancias.get)

        if clase_real == "classic" and prediccion == "classic":
            celda = "TP"
            TP += 1
        elif clase_real == "nonclassic" and prediccion == "nonclassic":
            celda = "TN"
            TN += 1
        elif clase_real == "nonclassic" and prediccion == "classic":
            celda = "FP"
            FP += 1
        else:
            celda = "FN"
            FN += 1

        filas_reporte.append([
            nombres[i],
            "Positive" if clase_real == "classic" else "Negative",
            round(distancias["classic"], 4),
            round(distancias["nonclassic"], 4),
            "Positive" if prediccion == "classic" else "Negative",
            celda
        ])

    columnas = [
        "Patron", "Clase_Real", "d_classic",
        "d_nonclassic", "Clase_Predicha", "Celda_MC"
    ]
    df_reporte = pd.DataFrame(filas_reporte, columns=columnas)
    df_reporte.to_csv("Resultados_Euclidiano_LOO_Nutt.csv", index=False)

    # Cálculo de medidas de desempeño
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    balanced_accuracy = (sensitivity + specificity) / 2
    precision = TP / (TP + FP)
    f1 = 2 / ((1 / precision) + (1 / sensitivity))
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    metricas = {
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Balanced Accuracy": balanced_accuracy,
        "Precision": precision,
        "F1": f1,
        "MCC": mcc
    }
    return df_reporte, metricas


# ---------------------------------------------------------
# BLOQUE PRINCIPAL
# ---------------------------------------------------------
if __name__ == "__main__":
    ruta_archivo = "Nutt (1).csv"

    X, y, nombres = cargar_nutt(ruta_archivo)
    df_res, metricas = evaluar_loo_euclidiano(X, y, nombres)

    print("=== TABLA DE RESULTADOS CLASIFICADOR EUCLIDIANO (LOO) ===")
    print(df_res.to_string(index=False))

    print("\n=== MATRIZ DE CONFUSIÓN ===")
    print(f"TP = {metricas['TP']} | FN = {metricas['FN']}")
    print(f"FP = {metricas['FP']} | TN = {metricas['TN']}")

    print("\n=== MEDIDAS DE DESEMPEÑO ===")
    print(f"a) Accuracy:          {metricas['Accuracy']:.4f} ({metricas['Accuracy'] * 100:.2f}%)")
    print(f"a) Sensitivity:       {metricas['Sensitivity']:.4f} ({metricas['Sensitivity'] * 100:.2f}%)")
    print(f"b) Specificity:       {metricas['Specificity']:.4f} ({metricas['Specificity'] * 100:.2f}%)")
    print(f"c) Balanced Accuracy: {metricas['Balanced Accuracy']:.4f} ({metricas['Balanced Accuracy'] * 100:.2f}%)")
    print(f"d) Precision:         {metricas['Precision']:.4f} ({metricas['Precision'] * 100:.2f}%)")
    print(f"e) F1:                {metricas['F1']:.4f} ({metricas['F1'] * 100:.2f}%)")
    print(f"e) MCC:               {metricas['MCC']:.4f}")