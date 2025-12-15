import numpy as np


"""

Funkcija, kuri įgyvendina daugiakanalio spredimo filtravimą:
Jeigu bent dviejuose kanaluose tame pačiame segmente buvo vienetai, tai tada tikrai įvyko priepuolis
Jeigu tik viename kanale vienetas, tikriname ar gretimuose segmentuose irgi yra vienetai. Jeigu nors vienas gretimas segmentas pažymėtas vientu, tada priepuolis įvyko.
Parametrai:
y_matrix - y klasės etikečių matrica
Grąžinama:
išfiltruotas y klasės vektorius

"""


def multichannel_decision_filtering(y_matrix):

    y_predicted_window = []

    for i in range(len(y_matrix)):

        if y_matrix[i].sum() >= 2:
            y_predicted_window.append(1)
        else:
            y_predicted_window.append(0)

    y_predicted_window = np.array(y_predicted_window)

    length_till_second_to_last_element = len(y_matrix) - 1

    for z in range(length_till_second_to_last_element):

        if y_matrix[z].sum() >= 1 and (
            (z > 0 and y_predicted_window[z - 1] == 1) or y_predicted_window[z + 1] == 1
        ):
            y_predicted_window[z] = 1

    return y_predicted_window


""" 
Atlieka necentralizuotą slankiojo vidurkio filtravimą
Parametrai:
signal - signalai
total_window_number - langų skaičius
channel_number - kanalų skaičius
values_count_to_change_current_value - kelių reikšmių vidurkiu bus pakeista esama reikšmę
Grąžinama:
necentralizuoto slankiojo vidurkio rezultatas

"""


def non_central_MAF(
    signal, total_window_number, channel_number, values_count_to_change_current_value
):

    windows_recreated_size = (
        total_window_number - values_count_to_change_current_value + 1
    )

    result = [
        [None for _ in range(channel_number)] for _ in range(windows_recreated_size)
    ]

    for w in range(windows_recreated_size):

        for ch in range(channel_number):

            start = w
            end = w + values_count_to_change_current_value
            window = signal[start:end, ch]
            result[w][ch] = np.mean(window)

    return result


""" 
Atlieka centralizuotą slankiojo vidurkio filtravimą
Parametrai:
signal - signalai
total_window_number - langų skaičius
values_count_to_change_current_value - kelių reikšmių vidurkiu bus pakeista esama reikšmę
Grąžinama:
centralizuoto slankiojo vidurkio rezultatas
"""


def central_MAF(signal, channel_number, values_count_to_change_current_value):

    result = []

    kernel = (
        np.ones(values_count_to_change_current_value)
        / values_count_to_change_current_value
    )

    for ch in range(channel_number):

        result.append(np.convolve(signal[:, ch], kernel, "same"))

    rez = np.array(result).transpose(1, 0)

    return rez
