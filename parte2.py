import math
import numpy as np
import pandas as pd

# 1. Cargar dataset Nutt (sin encabezado: 28 patrones x 1070 atributos + 1 clase)
df = pd.read_csv("Nutt (1).csv", header=None)
X = df.iloc[:, :-1].values.astype(float)
y = df.iloc[:, -1].values
n = len(X)
k = 27

# 2. Calcular matriz de distancias euclidianas (28 x 28)
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])

# 3. Validación Leave-One-Out con 27-NN
TP = TN = FP = FN = 0
tabla_desarrollo = []

for i in range(n):
    # Excluir el patrón de prueba i (quedan 27 patrones de entrenamiento)
    vecinos = [(j, dist_matrix[i, j], y[j]) for j in range(n) if j != i]
    vecinos_ordenados = sorted(vecinos, key=lambda item: item[1])
    k_vecinos = vecinos_ordenados[:k]

    votos_pos = sum(1 for _, _, clase in k_vecinos if clase == "classic")
    votos_neg = sum(1 for _, _, clase in k_vecinos if clase == "nonclassic")

    pred = "classic" if votos_pos > votos_neg else "nonclassic"
    real = y[i]

    if real == "classic" and pred == "classic":
        TP += 1
    elif real == "nonclassic" and pred == "nonclassic":
        TN += 1
    elif real == "nonclassic" and pred == "classic":
        FP += 1
    elif real == "classic" and pred == "nonclassic":
        FN += 1

    tabla_desarrollo.append({
        "Patron": f"pp{i+1}",
        "Clase_Real": "Positive" if real == "classic" else "Negative",
        "Vecino_1": f"pp{vecinos_ordenados[0][0]+1} ({vecinos_ordenados[0][1]:.4f})",
        "Vecino_27": f"pp{vecinos_ordenados[-1][0]+1} ({vecinos_ordenados[-1][1]:.4f})",
        "Votos_Positive": votos_pos,
        "Votos_Negative": votos_neg,
        "Clase_Asignada": "Positive" if pred == "classic" else "Negative",
    })

res_df = pd.DataFrame(tabla_desarrollo)
print(res_df.to_string(index=False))

# 4. Métricas de desempeño
total = TP + TN + FP + FN
accuracy = (TP + TN) / total
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
balanced_accuracy = (sensitivity + specificity) / 2
precision = TP / (TP + FP)
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
mcc = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

print(f"\nMatriz de confusión: TP={TP}, FN={FN}, FP={FP}, TN={TN}")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Sensitivity:       {sensitivity:.4f}")
print(f"Specificity:       {specificity:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"F1-score:          {f1:.4f}")
print(f"MCC:               {mcc:.4f}")