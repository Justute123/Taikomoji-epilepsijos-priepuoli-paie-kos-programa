import numpy as np
import math

from segmentation import split_to_windows
from labeling_seizures import fill_y_with_seizures

"""
Funkcija, kuri suranda atsitiktinius nepriepuolių langus, kurių skaičius sutampa su priepuolių langų skaičiumi.

Parametrai:
window_number -  langų skaičius
seizure_intervals - priepuolių laikų intervalai
seconds - po kiek sekundžių buvo suskirstyti segmentai
signals - EEG signalai
channel_number -  kanalų skaičius
one_window_signals_number - viename lange signalų duomenų skaičius
seizure_window_number_in_file - priepuolių langų skaičius faile

Grąžinama:
x_random_non_seizure -  nepriepuolių langų X vektorius
y_random_non_seizure - nepriepuolių langų klasės etikečių vektorius
random_non_seizure_indexes -  nepriepuolių langų indeksai



"""


def get_x_and_y_special_indexes_when_no_seizure(
    window_number,
    seizure_intervals,
    seconds,
    signals,
    channel_number,
    one_window_signals_number,
    seizure_window_number_in_file,
):

    y, seizures_number, non_seizures_number = fill_y_with_seizures(
        window_number, seizure_intervals, seconds
    )
    X = split_to_windows(
        signals, channel_number, window_number, one_window_signals_number
    )

    non_seizure_indexes = []

    for index in range(len(y)):

        if y[index] == 0:

            non_seizure_indexes.append(index)

    random_non_seizure_indexes = np.random.choice(
        non_seizure_indexes, size=seizure_window_number_in_file, replace=False
    )

    x_random_non_seizure = []
    for i in random_non_seizure_indexes:
        x_random_non_seizure.append(X[i])

    y_random_non_seizure = y[random_non_seizure_indexes]
    return x_random_non_seizure, y_random_non_seizure, random_non_seizure_indexes


"""
Funkcija, kuri suranda priepuolių langus, jų X vektorių, y klasės etikečių vektorių ir indeksus

Parametrai:
window_number -  langų skaičius
seizure_intervals - priepuolių laikų intervalai
seconds - po kiek sekundžių buvo suskirstyti segmentai
signals - EEG signalai
channel_number -  kanalų skaičius
one_window_signals_number - viename lange signalų duomenų skaičius

Grąžinama:
x_seizures -  priepuolių langų X vektorius
y_seizures -  priepuolių langų klasės etikečių vektorius
seizures_indexes -  priepuolių langų indeksai


"""


def get_x_and_y_special_indexes_when_seizure(
    window_number,
    seizure_intervals,
    seconds,
    signals,
    channel_number,
    one_window_signals_number,
):

    y, seizure_number, non_seizure_number = fill_y_with_seizures(
        window_number, seizure_intervals, seconds
    )
    X = split_to_windows(
        signals, channel_number, window_number, one_window_signals_number
    )

    seizure_indexes = []

    for index in range(len(y)):

        if y[index] == 1:

            seizure_indexes.append(index)

    x_seizures = []

    for i in seizure_indexes:

        x_seizures.append(X[i])

    y_seizures = y[seizure_indexes]

    return x_seizures, y_seizures, seizure_indexes


"""
    Funkcija, kuri sujungia X vertes nepriepuolių ir priepuolių (vyksta balansavimas) bei sujungia Y vertes nepriepuolių ir priepuolių.
    Parametrai:
    x_seizures -  x priepuolių vektorius
    x_non_seizures -  x nepriepuolių vektorius
    y_seizures - y priepuolių vektorius
    y_non_seizures - y nepriepuolių vektorius
    train_nr - treniravimui skirtų pacientų skaičius
    Graąžinama:
    x_concatinated - sujungti nepriepuolių ir priepuolių x
    y_conacitinated - sujungti nepriepuolių ir priepuolių y
    
 
"""


def conactinating_seizures_with_non_seizures(
    x_seizures, x_non_seizures, y_seizures, y_non_seizures, train_nr
):

    x_balanced_all = x_seizures[:train_nr] + x_non_seizures[:train_nr]
    y_balanced_all = y_seizures[:train_nr] + y_non_seizures[:train_nr]
    x_concatinated = np.vstack(x_balanced_all)
    y_concatinated = np.hstack(y_balanced_all)

    return x_concatinated, y_concatinated
