import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
import numpy as np
from feature_extraction import feature_calculation, dwt_coeffs
import matplotlib

matplotlib.use("QtAgg")

"""
Funkcija atvaizduoja ir išsaugo painiavos matricą. Nupiešia žalią matricą, rodo tik sveikus skaičius, nustato ašių pavadinimus, palieka 20% vietos kairėje.
Parametrai:
sumed_confusion_matrix - jau apskaičiuota painiavos matrica

"""


def plot_confusion_matrix(sumed_confusion_matrix):
    display = ConfusionMatrixDisplay(
        confusion_matrix=sumed_confusion_matrix,
        display_labels=["Ne priepuolis", "Priepuolis"],
    )
    display.plot(cmap="Greens", values_format="d")
    display.ax_.set_xlabel("Prognozuota reikšmė")
    display.ax_.set_ylabel("Tikroji reikšmė")
    plt.subplots_adjust(left=0.2)
    plt.title("Painiavos matrica")
    plt.savefig("final matrix", bbox_inches="tight")
    plt.show()


"""
Braižomas DWT grafikas, apskaičiuojami dwt koeficientai. Grafikas suskaidomas į 6 dažnio juostas (eilutes ir vieną stulpelį).
Paverčiama x ašy į keturių sekundžių segmentus.
Parametrai:
window - segmentas
channel_number - kanalų skaičius
window_number - langų skaičius


"""


def plot_dwt(window, channel_number, window_number):

    coefs = [[None for _ in range(channel_number)] for _ in range(window_number)]
    for w in range(window_number):
        for ch in range(channel_number):
            cA5, cD5, cD4, cD3, cD2, cD1 = dwt_coeffs(window[w][ch])
            coefs[w][ch] = [cA5, cD5, cD4, cD3, cD2, cD1]

    figure, axes = plt.subplots(6, 1, figsize=[14, 8])
    figure.suptitle("EEG signalo DWT koeficientai", fontweight="bold")
    figure.supxlabel("laikas (s)")
    figure.supylabel("elektrinis aktyvumas (V)")
    cf = coefs[752][20]
    for i in range(6):
        # reikia, kad x asys issidestytu vienodai graziai
        spaces = 4 / len(cf[i])
        time = np.arange(0, 4, spaces)
        axes[i].plot(time, cf[i], c="b")
        axes[i].set_xlim(0, 4)
        # skirtuku isdestymas x juostoj
        ticksSpaces = np.arange(0, 5, 1)
        axes[i].set_xticks(ticksSpaces)
        axes[i].ticklabel_format(style="plain", axis="y")

    axes[0].set_title("EEG")
    axes[1].set_title("d5")
    axes[2].set_title("d4")
    axes[3].set_title("d3")
    axes[4].set_title("d2")
    axes[5].set_title("d1")
    plt.subplots_adjust(hspace=1)
    plt.show()


"""
Nubraižomas EEG grafikas, surandama maksimali amplitudė, tam, kad vaizdas neužliptų vienas ant kito 
Parametrai:
data - EEG signalų duomenys
times - laikų vektorius
channel_names - kanalų pavadinimai
"""


def plot_EEG_graph(data, times, channel_names):

    maximal_ampltude = np.max(np.abs(data))
    plt.figure(figsize=(14, 8))

    for i in range(len(channel_names)):
        plt.plot(times, data[i] + i * maximal_ampltude, label=channel_names[i])

    plt.title("EEG signalas")
    plt.xlabel("laikas (s)")
    plt.ylabel("Amplitude (V)")
    plt.legend(loc="center right", fontsize=10)
    plt.tight_layout()
    plt.show()


"""
Nubrėžiama ROC kreivė ir apskaičiuojamas AUC
Parametrai:
y_true - teisingos y vertės
y_pred - prognozuojamos y vertės

"""


def plot_roc_graph(y_true, y_pred):

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC kreivė (AUC = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--", label="bazinė linija")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Klaidingai teigiamų priepuolių dažnis")
    plt.ylabel("Teisingai teigiamų priepuolių dažnis")
    plt.title("ROC kreivė epilepsijos priepuolių klasifikavimui")
    plt.legend()
    plt.show()


"""
Nupiešiamas sprendimo ribos grafikas. Randamos maximalios ir minimalios PCA1 ašys,
randamos maximalios ir minimalios PCA2 ašys, suskirtstoma į tinklelį (langelius)
paverčiamas masyvas į vektorius (x,y) ir sujungiami į stulpelius. Paskui nubrėžiamas grafikas su prognozuotom vertėm.
Parametrai:
model - SVM klasifikatorius
X - treniravimui skirti požymiai
y - treniravimui skirtos klasės etiketės
"""


def plot_decision_boundary(model, X, y):

    x_axis_min, x_axis_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    y_axis_min, y_axis_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    x_grid, y_grid = np.meshgrid(
        np.linspace(x_axis_min, x_axis_max, 500),
        np.linspace(y_axis_min, y_axis_max, 500),
    )

    predicted = model.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
    predicted = predicted.reshape(x_grid.shape)

    plt.figure()
    plt.contourf(
        x_grid,
        y_grid,
        predicted,
    )
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Nepriepuoliai (y=0)")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Priepuoliai (y=1)")
    print("Nepriepuoliai:", np.sum(y == 0))
    print("Priepuoliai:", np.sum(y == 1))
    plt.title("Linijinis SVM")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.show()
