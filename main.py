import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

# Diretório onde seus dados estão localizados
base_dir = 'D:/UFCA/Aprendizado de Maquina/Projeto/Detectar diseases em plantas/DataBase'

# Listas para armazenar histogramas de cores e rótulos
X = []  # Histogramas de cores
y = []  # Rótulos

# Iterar sobre os subdiretórios de treinamento (por exemplo, saudável, doenca1, doenca2, ...)
for class_name in os.listdir(os.path.join(base_dir, 'train')):
    class_dir = os.path.join(base_dir, 'train', class_name)

    # Iterar sobre as imagens no subdiretório
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)

        # Carregar a imagem
        image = cv2.imread(image_path)

        # Calcular o histograma de cores da imagem
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Adicionar o histograma de cores e o rótulo às listas
        X.append(hist)
        y.append(class_name)  # Use o nome do subdiretório como rótulo

# Converta rótulos em números inteiros
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Inicialize o validador cruzado com 10-fold estratificado
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Listas para armazenar as precisões de cada fold
accuracies = []
f1_macro_scores = []
f1_micro_scores = []
precision_scores = []

# Treinamento e teste para cada fold
for train_index, test_index in kf.split(X, y_encoded):
    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
    y_train, y_test = np.array(y_encoded)[train_index], np.array(y_encoded)[test_index]

    # Treinamento do modelo Random Forest nos histogramas de cores
    n_estimators = 100
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)

    # Previsões no conjunto de teste
    y_pred = rf_model.predict(X_test)

    # Calcular a precisão deste fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Calcular as métricas F1-macro e F1-micro
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_macro = report['macro avg']['f1-score']
    
    # Verificar se a chave 'micro avg' existe no dicionário
    if 'micro avg' in report:
        f1_micro = report['micro avg']['f1-score']
    else:
        f1_micro = 0.0  # Ou qualquer valor adequado

    f1_macro_scores.append(f1_macro)
    f1_micro_scores.append(f1_micro)

    # Calcular a métrica de precisão
    precision = report['macro avg']['precision']
    precision_scores.append(precision)

    # Mapear rótulos de volta para nomes de classes
    class_names = label_encoder.classes_
    y_test_names = [class_names[i] for i in y_test]
    y_pred_names = [class_names[i] for i in y_pred]

    # Imprimir se a planta está saudável ou doente com base nas previsões deste fold
    for i in range(len(X_test)):
        print(f'Imagem {i+1}: Rótulo Real - {y_test_names[i]}\nImagem {i+1}: Rótulo Previsto - {y_pred_names[i]}\n')

# Calcula a precisão média dos 10 folds
mean_accuracy = np.mean(accuracies)
mean_f1_macro = np.mean(f1_macro_scores)
mean_f1_micro = np.mean(f1_micro_scores)
mean_precision = np.mean(precision_scores)

print(f'Acuracia média do modelo (10-fold): {mean_accuracy:.2f}')
print(f'F1-macro médio do modelo (10-fold): {mean_f1_macro:.2f}')
print(f'F1-micro médio do modelo (10-fold): {mean_f1_micro:.2f}')
print(f'Precisão média do modelo (10-fold): {mean_precision:.2f}')

