import mne
import os
import numpy as np
from pywt import wavedec
from os import listdir
from os.path import isfile, join
import glob
import ntpath
import matplotlib

matplotlib.use("TkAgg")
import pickle
from pathlib import Path

from segmentation import split_to_windows
from preprocessing import preprocess_signals
from feature_extraction import feature_calculation, save_features_to_pkl_file
from clasification import calculate_SVM
from patients_info import get_patient_details
from balancing import (
    get_x_and_y_special_indexes_when_seizure,
    get_x_and_y_special_indexes_when_no_seizure,
    conactinating_seizures_with_non_seizures,
)
from labeling_seizures import (
    fill_y_with_seizures,
    formating_seizure_intervals,
    calculate_number_of_seizures,
)

"""
Funkcija, kuri išanalizuoja aplanko turinį: surenka EDF ir teksinio failo kelius ir pavadinimus.
Naudojantis jais atliekamas EDF failo informacijos surinkimas, išankstinis apdorojimas, segmentavimas, priepuolių sužymėjimas (treniravimui), balansavimas, apskaičiuojami požymiai,
pasinaudojus subalansuotais duomenimis
Parametrai:
seconds - po kiek laiko segmentai suskirtomi
one_patient_path - vieno paciento aplanko kelias
Grąžinama:
x_test - x požymių vektorius testavimui
y_test - y klasės etikečių vektorius testavimui
x_seizure - x požymių vektorius (tik priepuoliai)
y_seizure - y klasės etikečių vektorius (tik priepuolai)
x_non_seizure - x požymių vektorius (nepriepuoliai)
y_non_seizure - y klasės etikečių vektorius (nepriepuoliai)
ones -  vektrius užpilytas priepuolais
zeros - vektorius užpildytas ne priepuoliais
edf_files_paths - EDF failų keliai
channel_number - kanalų skaičius
seizures - priepuolių skaičius
cases_number - atvejų skaičius
window_number - langų skaičius
sample_size - duomenų taškų skaičius

"""


def load_one_patient(seconds, one_patient_path):

    total_seizure_windows = 0
    total_non_seizure_windows = 0
    cases_number = 0
    edf_file_names = []
    txt_file_names = []
    y_test = []
    x_test = []
    x_seizure = []
    y_seizure = []
    x_non_seizure = []
    y_non_seizure = []
    seizures = []
    seizure_list = []
    non_seizure_list = []

    only_files = [
        f for f in listdir(one_patient_path) if isfile(join(one_patient_path, f))
    ]

    txt_file_paths = glob.glob(join(one_patient_path, "*.txt"))
    for f in txt_file_paths:

        txt_file_names.append(ntpath.basename(f))

    edf_files_paths = glob.glob(join(one_patient_path, "*.edf"))
    for f in edf_files_paths:

        cases_number = cases_number + 1
        edf_file_names.append(ntpath.basename(f))

        seizure_intervals = formating_seizure_intervals(
            txt_file_paths, ntpath.basename(f)
        )
        (
            raw,
            frequency,
            sample_size,
            one_window_signals_number,
            duration_all_signal,
            window_number,
            channel_number,
        ) = get_patient_details(f, seconds)

        signals = preprocess_signals(
            raw, notch_freq=50, band_low_freq=0.5, band_high_freq=120
        )

        window = split_to_windows(
            signals, channel_number, window_number, one_window_signals_number
        )

        feature_vector_all_segments = feature_calculation(
            signals, window, frequency, channel_number, seconds, duration_all_signal
        )

        x_test.append(feature_vector_all_segments)
        y, ones, zeros = fill_y_with_seizures(window_number, seizure_intervals, seconds)
        seizures_number = calculate_number_of_seizures(y)

        seizures.append(seizures_number)
        y_test.append(y)
        seizure_list.append(ones)
        non_seizure_list.append(zeros)

        x_seizures_balanced, y_seizures_balanced, seizures_indexes_balanced = (
            get_x_and_y_special_indexes_when_seizure(
                window_number,
                seizure_intervals,
                seconds,
                signals,
                channel_number,
                one_window_signals_number,
            )
        )
        seizures_windows_number = int(np.sum(y))
        total_seizure_windows = total_seizure_windows + seizures_windows_number
        feature_vector_seizure_segments = feature_calculation(
            signals,
            x_seizures_balanced,
            frequency,
            channel_number,
            seconds,
            duration_all_signal,
            seizures_indexes_balanced,
        )
        x_seizure.append(feature_vector_seizure_segments)
        y_seizure.append(np.repeat(y_seizures_balanced, channel_number))

        x_non_seizure_balanced, y_non_seizure_balanced, non_seizures_indexes = (
            get_x_and_y_special_indexes_when_no_seizure(
                window_number,
                seizure_intervals,
                seconds,
                signals,
                channel_number,
                one_window_signals_number,
                seizures_windows_number,
            )
        )
        feature_vector_non_seizure_segments = feature_calculation(
            signals,
            x_non_seizure_balanced,
            frequency,
            channel_number,
            seconds,
            duration_all_signal,
            non_seizures_indexes,
        )

        non_seizures_windows_number = int(np.sum(y_non_seizure_balanced == 0))
        total_non_seizure_windows = (
            total_non_seizure_windows + non_seizures_windows_number
        )
        x_non_seizure.append(feature_vector_non_seizure_segments)
        y_non_seizure.append(np.repeat(y_non_seizure_balanced, channel_number))

    return (
        x_test,
        y_test,
        x_seizure,
        y_seizure,
        x_non_seizure,
        y_non_seizure,
        ones,
        zeros,
        edf_files_paths,
        channel_number,
        seizures,
        cases_number,
        window_number,
        sample_size,
    )


"""
Apdoroja visus pacientus ir išsaugo reikalingą informaciją į požymių failą
Parametrai:
seconds - po kiek laiko segmentai suskirtomi
"""


def load_all_patients(seconds):

    patient_index = 1
    root = "/home/justina/Documents/EEG_CHANGED"
    x_test = []
    y_test = []
    x_test_all = []
    y_test_all = []
    x_train_seizures = []
    y_train_seizures = []
    x_train_non_seizures = []
    y_train_non_seizures = []
    all_patient_seizures = []
    channels_per_patient = []
    windows_per_patient = []
    one_list = []
    zero_list = []
    edf_paths_list = []
    sample = []

    for one_patient_folder in sorted(os.listdir(root)):
        print("PATIENT OF INDEX:  ", patient_index)
        one_patient_path = os.path.join(root, one_patient_folder)
        (
            x_test,
            y_test,
            x_seizure,
            y_seizure,
            x_non_seizure,
            y_non_seizure,
            ones,
            zeros,
            edf_files_paths,
            channel_number,
            seizures,
            cases_number,
            window_number,
            sample_size,
        ) = load_one_patient(seconds, one_patient_path)
        channels_per_patient.append(channel_number)
        windows_per_patient.append(window_number)
        sample.append(sample_size)

        x_test_all.append(x_test[0])
        y_test_all.append(y_test[0])
        x_train_seizures.append(x_seizure[0])
        y_train_seizures.append(y_seizure[0])
        x_train_non_seizures.append(x_non_seizure[0])
        y_train_non_seizures.append(y_non_seizure[0])
        one_list.append(ones)
        zero_list.append(zeros)
        edf_paths_list.append(edf_files_paths)
        all_patient_seizures.append(seizures[0])
        patient_index = patient_index + 1

    patients_number = len(all_patient_seizures)
    x_final_15, y_final_15 = conactinating_seizures_with_non_seizures(
        x_train_seizures,
        x_train_non_seizures,
        y_train_seizures,
        y_train_non_seizures,
        train_nr=15,
    )

    save_features_to_pkl_file(
        x_test_all,
        y_test_all,
        x_final_15,
        y_final_15,
        channels_per_patient,
        all_patient_seizures,
        patients_number,
        windows_per_patient,
    )


"""

Užkraunamas požymių failas ir ši informacija yra pateikiama SVM modelio klasifikatoriui

"""


def Main():

    path = "/home/justina/Documents/EEG-darbas/EEG-darbas/KEISTAS/Projektas/featuresChanged.pkl"

    obj = Path(path)

    if obj.exists():

        with open("featuresChanged.pkl", "rb") as f:
            data_15_changed = pickle.load(f)
            x_test_15_changed = data_15_changed["XTest"]
            y_test_15_changed = data_15_changed["yTest"]
            x_train_15_changed = data_15_changed["XTrain"]
            yTrain15Changed = data_15_changed["yTrain"]
            channel_number_15_changed = data_15_changed["channelNumber"]
            seizures_15_changed = data_15_changed["seizures"]
            patients_number_15_changed = data_15_changed["patientsNumber"]
            window_number_15_changed = data_15_changed["windowNumber"]

        calculate_SVM(
            x_test=x_test_15_changed,
            y_test=y_test_15_changed,
            x_train=x_train_15_changed,
            y_train=yTrain15Changed,
            channel_number=channel_number_15_changed,
            patient_number=patients_number_15_changed,
            window_number=window_number_15_changed,
            train_nr=15,
        )
        # else:
        # load_all_patients(seconds=4)
        # Main()


Main()
