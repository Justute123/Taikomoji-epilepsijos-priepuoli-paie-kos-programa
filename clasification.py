from sklearn.svm import SVC
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("TkAgg")
from sklearn.svm import LinearSVC
import joblib


from postprocessing import multichannel_decision_filtering, non_central_MAF, central_MAF
from labeling_seizures import calculate_number_of_seizures
from utils import plot_confusion_matrix

"""
SVM Treniravimui skirta funkcija, kurioje išsaugomas ištreniruotas SVM modelis
Parametrai:
x_train - treniravimui skirtų pacientų požymių vektorius
y_train - treniravimui skirtų pacientų klasės etikečių y vektorius
"""


def training(x_train, y_train):

    scaler = StandardScaler()

    # svc_clf = SVC(probability=True)
    svc_clf = SVC(class_weight={0: 2, 1: 1}, C=1, gamma=0.001, probability=True)
    # svc_clf = LinearSVC()
    # svc_clf = LinearSVC(dual=False, max_iter=10000, class_weight={0: 2, 1: 1})

    np.set_printoptions(threshold=np.inf)

    x_train_scaled = scaler.fit_transform(x_train)

    svc_clf.fit(x_train_scaled, y_train)

    model_filename = "seizureFile.sav"
    joblib.dump(svc_clf, model_filename)
    scaler_filename = "scalerFile.sav"
    joblib.dump(scaler, scaler_filename)


"""
Šioje funkcijoje gaunami prognozuojamų priepuolių rezultatai prieš galutinių rezultatų apdorojimą

Parametrai:
x_test_scaled - standartizuoti treniravimui skirti duomenys
window_number - langų skaičius
channel_number - kanalų skaičius
svc_clf - apmokytas klasifikavimo modelis
i -  paciento indeksas sąraše
Grąžinamos:
y modelio prognozuojamos vertės

"""


def prepostprocessing_testing(
    x_test_scaled, window_number, channel_number, svc_clf, i=None
):
    if i is not None:

        window_n = window_number[i]
        channel_n = channel_number[i]

    else:

        window_n = window_number
        channel_n = channel_number

    y_predicted_before_reshaping = svc_clf.predict(x_test_scaled)
    y_matrix = y_predicted_before_reshaping.reshape(window_n, channel_n)
    y_predicted_after_reshaping = (y_matrix.sum(axis=1) >= 1).astype(int)

    return y_predicted_after_reshaping


"""
Šioje funkcijoje gaunami prognozuojamų priepuolių rezultatai po galutinių rezultatų apdorojimo

Parametrai:
x_test_scaled - standartizuoti treniravimui skirti duomenys
window_number - langų skaičius
channel_number - kanalų skaičius
svc_clf - apmokytas klasifikavimo modelis
i -  paciento indeksas sąraše
Grąžinamos:
y modelio prognozuojamos vertės

"""


def postprocesing_testing(
    x_test_scaled, window_number, channel_number, svc_clf, i=None
):

    if isinstance(window_number, (list)):

        window_n = window_number[i]
        channel_n = channel_number[i]

    else:

        window_n = window_number
        channel_n = channel_number

    raw_signal = svc_clf.decision_function(x_test_scaled)
    raw_matrix = raw_signal.reshape(window_n, channel_n)
    MAF = central_MAF(raw_matrix, channel_n, values_count_to_change_current_value=3)

    y_to_binary = (np.array(MAF) >= 0).astype(int)
    y_predicted_after_reshaping = multichannel_decision_filtering(y_to_binary)

    return y_predicted_after_reshaping


"""
Šioje funkcijoje yra atliekamas svm modelio testavimas
Parametrai:
x_test_scaled - standartizuoti treniravimui skirti duomenys
window_number - langų skaičius
channel_number - kanalų skaičius
svc_clf - apmokytas klasifikavimo modelis
i -  paciento indeksas sąraše
scaler - duomenų standartizavimo objektas
Grąžinamos:
y modelio prognozuojamos vertės

"""


def testing(x_test, window_number, channel_number, scaler, svc_clf, i=None):

    if i is not None:

        i = i

    else:

        i = 0

    x_test_per_patient = x_test[i]

    x_test_scaled = scaler.transform(x_test_per_patient)

    y_predicted_after_reshaping = postprocesing_testing(
        x_test_scaled, window_number, channel_number, svc_clf, i=i
    )

    return y_predicted_after_reshaping


"""
Funkcijoje yra apskaičiuojamos modelio vertinimo metrikos
Parametrai:
TP - teisingai diagnozuoti priepuoliai
FP - klaidingai diagnozuoti priepuoliai, nors tai buvo nepriepuoliai
TN - teisingai diagnozuoti nepriepuoliai
FN - klaidingai diagnozuoti nepriepuoliai
Grąžinamas:
jautrumas, speocefiškumas, tikslumas, preciziškumas, klaidingai teigiamŲ atvejŲ daŽnis ir F1

"""


def calculate_metrics(TP, FP, FN, TN):

    if (TP + FN) > 0:

        sensitivity = TP / (TP + FN)

    else:

        sensitivity = 0

    if (TN + FP) > 0:

        specificity = TN / (TN + FP)

    else:

        specificity = 0.0

    if (TP + TN + FP + FN) > 0:

        accurancy = (TP + TN) / (TP + TN + FP + FN)

    else:

        accurancy = 0.0

    if (TP + FP) > 0:

        precision = TP / (TP + FP)

    else:

        precision = 0

    if (sensitivity + precision) > 0:

        Fscore = 2 * (precision * sensitivity / (precision + sensitivity))

    else:

        Fscore = 0

    if (FP + TN) > 0:

        fpr = FP / (FP + TN)

    else:

        fpr = 0

    return sensitivity, specificity, accurancy, precision, Fscore, fpr


"""
Išvedamos metrikos
Parametrai:
TP - teisingai diagnozuoti priepuoliai
FP - klaidingai diagnozuoti priepuoliai, nors tai buvo nepriepuoliai
TN - teisingai diagnozuoti nepriepuoliai
FN - klaidingai diagnozuoti nepriepuoliai
"""


def check_metrics(TP, FP, FN, TN):

    sensitivity, specificity, accurancy, precision, Fscore, fpr = calculate_metrics(
        TP, FP, FN, TN
    )

    print("Test accurancy: ", round(accurancy * 100, 2), "%")
    print("Test sensitivity: ", round(sensitivity * 100, 2), "%")
    print("Test specificity: ", round(specificity * 100, 2), "%")
    print("Test precision: ", round(precision * 100, 2), "%")
    print("test Fscore: ", round(Fscore * 100, 2), "%")
    print("test fpr: ", round(fpr * 100, 2), "%")


"""
Šioje funkcijoje atliekama SVM modelio validacija
Parametrai:
i - testuojamo paciento indeksas
y_test_per_patient - testavimui skirtas pacientas 
redicted_after_reshaping - y modelio prognozuojamos vertės
TP - teisingai diagnozuoti priepuoliai
FP - klaidingai diagnozuoti priepuoliai, nors tai buvo nepriepuoliai
TN - teisingai diagnozuoti nepriepuoliai
FN - klaidingai diagnozuoti nepriepuoliai

"""


def validation(i, y_test_per_patient, y_predicted_after_reshaping, TP, FP, FN, TN):

    # evaluation
    print("Patient: ", i)
    print("positives in test y:", calculate_number_of_seizures(y_test_per_patient))
    print(
        "positives in predicted y:",
        calculate_number_of_seizures(y_predicted_after_reshaping),
    )

    check_metrics(TP, FP, FN, TN)


"""

Funkcijoje atliekamas SVM modelio treniravimas, testavimas, validavimas, gaunama painiavos matrica
x_test - testavimui skirtas x požymių vektorius
y_test - testavimui skirtas y klasės etikečių vektorius
x_train - treniravimui skirtas x požymių vektorius
y_train - treniravimui skirtas y klasės etikečių vektrois
channel_number - kanalų skaičius
patient_number - pacientų skaičius
window_number - langų skaičius
train_nr - pacientų skaičius, kurie bus skirti treniravimui

"""


def calculate_SVM(
    x_test,
    y_test,
    x_train,
    y_train,
    channel_number,
    patient_number,
    window_number,
    train_nr,
):

    sum_TN = 0
    sum_FN = 0
    sum_FP = 0
    sum_TP = 0

    training(x_train, y_train)
    svc_clf = joblib.load("seizureFile.sav")
    scaler = joblib.load("scalerFile.sav")

    for i in range(train_nr, patient_number):

        y_test_per_patient = y_test[i]
        y_predicted_after_reshaping = testing(
            x_test, window_number, channel_number, scaler, svc_clf, i)
        

        confusion_matrix_test = confusion_matrix(
            y_test_per_patient[: len(y_predicted_after_reshaping)],
            y_predicted_after_reshaping,
            labels=[0, 1],
        )

        print("Test confusion matrix:\n", confusion_matrix_test)

        TN = confusion_matrix_test[0, 0]
        FP = confusion_matrix_test[0, 1]
        FN = confusion_matrix_test[1, 0]
        TP = confusion_matrix_test[1, 1]

        sum_FN = sum_FN + FN
        sum_TN = sum_TN + TN
        sum_FP = sum_FP + FP
        sum_TP = sum_TP + TP

        validation(i, y_test_per_patient, y_predicted_after_reshaping, TP, FP, FN, TN)

    summed_confusion_matrix = np.matrix([[sum_TN, sum_FP], [sum_FN, sum_TP]])

    print("summed confusion matrix: ", summed_confusion_matrix)
    print("Overall model metrics: ")
    check_metrics(sum_TP, sum_FP, sum_FN, sum_TN)
    plot_confusion_matrix(summed_confusion_matrix)
