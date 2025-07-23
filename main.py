import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from skimage.feature import local_binary_pattern
from scipy.stats import wilcoxon

# Diretório onde as amostras estão localizadas
data_dir = 'D:/UFCA/Aprendizado de Maquina/Projeto/Detectar diseases em plantas/DataBase'

# Função para extrair recursos LBP da imagem
def extract_lbp_features(image):
    radius = 3
    n_points = 24
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-5)  # Normalização do histograma
    return hist

# Listas para armazenar dados de treinamento (histogramas de cores, recursos LBP) e rótulos de treinamento
X_hist = []  # Histogramas de cores de treinamento
X_lbp = []  # Recursos LBP de treinamento
y = []  # Rótulos de treinamento

# Iterar sobre as amostras no diretório
for class_name in os.listdir(os.path.join(data_dir, 'train')):
    class_dir = os.path.join(data_dir, 'train', class_name)

    # Iterar sobre as imagens no diretório
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)

        # Carregar a amostra
        sample = cv2.imread(image_path)

        # Calcular o histograma de cores da amostra
        sample_hist = cv2.calcHist([sample], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        sample_hist = cv2.normalize(sample_hist, sample_hist).flatten()

        # Adicionar o histograma de cores da amostra e o rótulo da amostra às listas
        X_hist.append(sample_hist)
        y.append(class_name)  # Use o nome do subdiretório como rótulo da amostra

        # Lê a amostra em escala de cinza para recursos LBP
        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        sample_lbp_features = extract_lbp_features(sample_gray)
        X_lbp.append(sample_lbp_features)

# Converte rótulos de amostras em números inteiros
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Divide os dados em treinamento (80%) e teste (20%)
X_train_hist, X_test_hist, y_train_encoded, y_test_encoded = train_test_split(X_hist, y_encoded, test_size=0.2, random_state=42)
X_train_lbp, X_test_lbp, y_train_encoded_lbp, y_test_encoded_lbp = train_test_split(X_lbp, y_encoded, test_size=0.2, random_state=42)

# Inicializa o validador cruzado com 10-fold estratificado
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Função para treinar e testar um modelo
def train_and_test(X, model, model_name, X_test=None, y_test=None):
    # Listas para armazenar as métricas e matrizes de confusão de cada fold
    accuracies = []
    f1_macro_scores = []
    f1_micro_scores = []
    precision_scores = []
    confusion_matrices = []

    # Treinamento e teste para cada fold
    for train_index, test_index in kf.split(X, y_train_encoded): 
        X_train, X_val = np.array(X)[train_index], np.array(X)[test_index]
        y_train, y_val = np.array(y_train_encoded)[train_index], np.array(y_train_encoded)[test_index]

        # Treinamento do modelo nos recursos de treinamento
        model.fit(X_train, y_train)

        if X_test is not None and y_test is not None:
            # Teste do modelo nos dados de teste
            y_pred = model.predict(X_val)

            # Calcular a precisão deste fold
            accuracy = accuracy_score(y_val, y_pred)
            accuracies.append(accuracy)

            # Calcular as métricas F1-macro e F1-micro
            report = classification_report(y_val, y_pred, output_dict=True)
            f1_macro = report['macro avg']['f1-score']

            if 'micro avg' in report:
                f1_micro = report['micro avg']['f1-score']
            else:
                f1_micro = 0.0

            f1_macro_scores.append(f1_macro)
            f1_micro_scores.append(f1_micro)

            # Calcular a métrica de precisão
            precision = report['macro avg']['precision']
            precision_scores.append(precision)

            # Calcular a matriz de confusão deste fold
            cm = confusion_matrix(y_val, y_pred)
            confusion_matrices.append(cm)

    if X_test is not None and y_test is not None:
        # Calcula as métricas médias com desvio padrão
        mean_accuracy = np.mean(accuracies)
        mean_f1_macro = np.mean(f1_macro_scores)
        mean_f1_micro = np.mean(f1_micro_scores)
        mean_precision = np.mean(precision_scores)

        std_accuracy = np.std(accuracies)
        std_f1_macro = np.std(f1_macro_scores)
        std_f1_micro = np.std(f1_micro_scores)
        std_precision = np.std(precision_scores)

        # Imprime as métricas médias com desvio padrão
        print(f'Métricas para {model_name}:\n')
        print(f'Acurácia média (10-fold): {mean_accuracy:.2f} ± {std_accuracy:.2f}')
        print(f'F1-macro médio (10-fold): {mean_f1_macro:.2f} ± {std_f1_macro:.2f}')
        print(f'F1-micro médio (10-fold): {mean_f1_micro:.2f} ± {std_f1_micro:.2f}')
        print(f'Precisão média (10-fold): {mean_precision:.2f} ± {std_precision:.2f}\n')

        # Imprime as matrizes de confusão de cada fold
        for i, cm in enumerate(confusion_matrices):
            print(f'Matriz de Confusão - Fold {i+1}:\n{cm}\n')

# Divide os dados em treinamento (80%) e teste (20%) uma única vez
X_train_hist, X_test_hist, y_train_encoded, y_test_encoded = train_test_split(X_hist, y_encoded, test_size=0.2, random_state=42)
X_train_lbp, X_test_lbp, y_train_encoded_lbp, y_test_encoded_lbp = train_test_split(X_lbp, y_encoded, test_size=0.2, random_state=42)

# Treina e testa o modelo Random Forest com recursos de histogramas de cores
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_test(X_train_hist, rf_model, 'Random Forest com Histograma de Cores', X_test=X_test_hist, y_test=y_test_encoded)

# Treina e testa o modelo Random Forest com recursos LBP
rf_model_lbp = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_test(X_train_lbp, rf_model_lbp, 'Random Forest com LBP', X_test=X_test_lbp, y_test=y_test_encoded_lbp)

# Inicializa o classificador SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# Testa os modelos nos dados de teste
train_and_test(X_test_hist, svm_model, 'SVM com Histograma de Cores (Teste)', X_test=X_test_hist, y_test=y_test_encoded)
train_and_test(X_test_hist, rf_model, 'Random Forest com Histograma de Cores (Teste)', X_test=X_test_hist, y_test=y_test_encoded)
train_and_test(X_test_lbp, rf_model_lbp, 'Random Forest com LBP (Teste)', X_test=X_test_lbp, y_test=y_test_encoded)

# Adiciona o teste de Wilcoxon para comparar as três configurações
# Define as listas de métricas para cada configuração
accuracies_histogram_rf = [1.00, 1.00, 0.00, 1.00]
accuracies_lbp_rf = [0.98, 0.98, 0.00, 0.98]
accuracies_svm = ["Histograma + SVM"]

# Realiza o teste de Wilcoxon usando as métricas obtidas
_, p_value_accuracies_rf_lbp = wilcoxon(accuracies_histogram_rf, accuracies_lbp_rf)
_, p_value_accuracies_rf_svm = wilcoxon(accuracies_histogram_rf, accuracies_svm)
_, p_value_accuracies_lbp_svm = wilcoxon(accuracies_lbp_rf, accuracies_svm)

# Imprime os resultados
print(f'p-value para Acurácia entre "Histograma + Random Forest" e "LBP + Random Forest": {p_value_accuracies_rf_lbp}')
print(f'p-value para Acurácia entre "Histograma + Random Forest" e "Histograma + SVM": {p_value_accuracies_rf_svm}')
print(f'p-value para Acurácia entre "LBP + Random Forest" e "Histograma + SVM": {p_value_accuracies_lbp_svm}')
