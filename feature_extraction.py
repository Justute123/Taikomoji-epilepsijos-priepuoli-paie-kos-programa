import matplotlib

matplotlib.use("TkAgg")  # arba Qt5Agg
import matplotlib.pyplot as plt
import numpy as np
from pywt import wavedec
import math
import pickle

"""

Funkcija, kuri suskaido gautą signalą į tam tikrus dažnius.
Parametrai:
signal_vector - signalo segmentas
Grąžinama: 6 koeficientų masyvai tam tikruose dažniuose, kurie bus panaudoti požymių skaičiavimui.

"""


def dwt_coeffs(signal_vector):
    return wavedec(signal_vector, "db4", level=5)


"""

Funkcija, kuri apskaičiuoja absoliučiąją energiją tam tikroje dažnių skalėje.
Parametrai:
dwt_coeffs - dwt koeficientai tam tikroje skalėje.
number_of_dwt_coeffs - dwt koeficientų skaičius tam tikroje skalėje.
frequency - mėginių surinkimo dažnis.
Grąžinama: 
result - absoliuti energija tam tikroje dažnių skalėje.

"""


def absolute_energy(dwt_coeffs, number_of_dwt_coeffs, frequency):

    sampling_interval = 1 / frequency
    rate = sampling_interval / number_of_dwt_coeffs
    result = np.sum((dwt_coeffs * dwt_coeffs) * rate)

    return result


"""

Funckija, kuri skaičiuoja santykinę energiją.
Parametrai:
absolute_energy_at_scale - absoliuti energija tam tikroje skalėje.
sum_of_absolute_energy - absoliučių energijų suma skalėje D3, D4, D5.
Grąžinama:
santykinė energija tam tikroje skalėje

"""


def relative_energy(absolute_energy_at_scale, sum_of_absolute_energy):
    return absolute_energy_at_scale / sum_of_absolute_energy


"""
Funkcija randa indeksus, kuriuose vertės kerta 0 ašį.
Parametrai:
coefs - dwt koeficientai.
Grąžinama:
indeksai, kur kertamas 0 ašis.
"""


def find_zero_crossings(coefs):
    zero_crossings = np.where(np.diff(np.sign(coefs)) != 0)[0]
    return zero_crossings


"""
Funkcija, kuri suskirsto dwt koeficientus į segmentus, kur yra kertama nulio ašis.
(šie segmentai yra persidengiantys per vieną koeficientą, tam, kad nebūtų ptrarandama informacija).
Parametrai:
coeffs - dwt koeficientai
zero_crossing_indexes - indeksai, kuriuose kerta nulio ašį.
Grąžinama:
half_wave_segment - pusbangės segmentas.

"""


def find_half_wave_segments(coeffs, zero_crossing_indexes):

    half_wave_segment = []
    start = 0
    end = zero_crossing_indexes[0] + 2

    for i in range(len(zero_crossing_indexes) - 1):

        half_wave_segment.append(coeffs[start:end])
        start = zero_crossing_indexes[i] + 1
        end = zero_crossing_indexes[i + 1] + 2

    half_wave_segment.append(coeffs[start:])

    return half_wave_segment


"""

Funkcija, kuri apskaičiuoja pusbangių amplitudes.
Parametrai:
half_wave_segment - pusbangės segmentas.
Grąžinama:
half_wave_amplitudes - pusbangių amtplitudės.

"""


def find_half_wave_amplitudes(half_wave_segment):

    half_wave_amplitudes = []

    for el in half_wave_segment:

        max = np.max(el)
        min = np.min(el)
        half_wave_amplitudes.append(max - min)

    return half_wave_amplitudes


"""

Funkcija, kuri apskaičiuoja epochos abosliučiąją amplitudę
Parametrai:
N - pusbangių amplitudžių skaičius
half_wave_amplitudes - pusbangių ampltidės
Grąžinama:
amplitude_epoch - absoliuti amplitudė

"""


def absolute_amplitude_in_epoch(N, half_wave_amplitudes):

    total_amplitude = np.sum(half_wave_amplitudes)
    amplitude_epoch = total_amplitude / N

    return amplitude_epoch


"""

Funkcija, kuri vykdo apskaičiavimus, norint surasti absoliučiąją amplitudę
Parametrai:
coeff - dwt koeficientai
Grąžinama:
amplitude - absoliuti ampltiudė

"""


def calculate_amplitudes(coeff):

    zero_crossings = find_zero_crossings(coeff)

    half_wave_segment = find_half_wave_segments(coeff, zero_crossings)

    half_wave_amplitudes = find_half_wave_amplitudes(half_wave_segment)

    amplitude = absolute_amplitude_in_epoch(
        len(half_wave_amplitudes), half_wave_amplitudes
    )

    return amplitude


"""
Funkcija, kuri suskaido siganalą į foninių segmentų atkarpas. Jos bus naudojamos apskaičiuoti santykinę amplitudę
Parametrai:
channel_number - kanalų skaičius
frequency - mėginių surinkimo dažnis
duration_all_signals - visų segmentų trukmė
duration - vieno fono segmento trukmė
gap - tarpas tarp segmentų (sec)
Grąžinama:
windows - fono segmentai
indexes - fono segmentų indeksai
window_nr - fono segmentų skaičius
"""


def split_to_windows_choseen_size_for_amplitudes_background_calculation(
    channel_number, frequency, signals, duration_all_signals, duration, gap
):
    start = 0
    end = int(duration * frequency)

    window_nr = int(duration_all_signals / (duration + gap))

    windows = [[None for _ in range(channel_number)] for _ in range(window_nr)]
    indexes = []

    for w in range(window_nr):

        for ch in range(channel_number):

            windows[w][ch] = signals[ch, start:end]

        indexes.append((start, end))

        background_duration = int((duration + gap) * frequency)
        start = start + background_duration
        end = start + int(duration * frequency)

    return windows, indexes, window_nr


"""
Apskaičiuojama foninė absoliučioji amplitudė, kuri bus skirta apskaičiuoti santykinei amplitudei.
Parametrai:
channel_number - kanalų skaičius.
windows - foninių segmentų atkarpos.
Grąžinamas sąrašas, kuriame yra d3, d4, d5 foninė amplitudė.

"""


def amplitude_background(channel_number, windows):

    background = []
    for w in range(len(windows)):
        for ch in range(channel_number):
            cA5, cD5, cD4, cD3, cD2, cD1 = dwt_coeffs(windows[w][ch])

            amplitude_d3_background = calculate_amplitudes(cD3)
            amplitude_d4_background = calculate_amplitudes(cD4)
            ampltiude_d5_background = calculate_amplitudes(cD5)

            background.append(
                [
                    amplitude_d3_background,
                    amplitude_d4_background,
                    ampltiude_d5_background,
                ]
            )

    return background


"""

Apskaičiuojamas variacijos koeficientas
Parametrai:
coeffs - dwt koeficientai
N - dwt koeficientų skaičius
Grąžinama:
variacijos koeficientų vertė

"""


def coefficient_variation(coeffs, N):

    mean_value = (1 / N) * np.sum(coeffs)

    standart_daviation = math.sqrt(
        (1 / N) * np.sum((coeffs - mean_value) * (coeffs - mean_value))
    )

    coef_variation = math.pow(standart_daviation, 2) / math.pow(mean_value, 2)

    return coef_variation


"""

Apskaičiuojamas svyravimo indeksas
Parametrai:
coeffs - dwt koeficientai
N - dwt koeficientų skaičius
Grąžinama:
svyravimo indekso vertė

"""


def fluctuation_index(coeff, N):

    fluctuation_ind = (1 / N) * np.sum(abs(np.diff(coeff, prepend=0)))
    return fluctuation_ind


"""
Funkciją apskaičiuoja foninio segmento indeksą
Parametrai:
time - signalo segmento pradžios laikas
window_nr - foninių atkarpų skaičius
"""


def find_background_amplitude_index(time, window_nr):
    period = 180

    if time <= (120 + 60):

        index = 0
    else:

        index_a = time - (120 + 60)
        index = int(index_a / period)

    index = min(index, window_nr - 1)

    return index


""" 

Funkcija apskaičiuoja svarbius segmentų požymius: santykinę energija, amplitudę, variacijos koeficientą ir svyravimo indeksą.
Parametrai:
signals - signalai, kuriems buvo taikytas išankstinis apdorojimas
windows - segmentai
frequency - mėginių surinkimo dažnis
channel_number -  kanalų skaičius
seconds - sekundės po kiek sec. buvo suskirstyta į segmentus
duration_all_signal - viso signalo trukmė sekundėmis
seizure_or_non_seizure_segment_indexes - priepuolinių/nepriepuolinių segmentų indeksai
Grąžinama:
požymių vektorius

"""


def feature_calculation(
    signals,
    windows,
    frequency,
    channel_number,
    seconds,
    duration_all_signal,
    seizure_or_non_seizure_segment_indexes=None,
):
    features = []

    background_windows, indexes, window_nr = (
        split_to_windows_choseen_size_for_amplitudes_background_calculation(
            channel_number,
            frequency,
            signals,
            duration_all_signal,
            duration=120,
            gap=60,
        )
    )

    background_amplitude = amplitude_background(channel_number, background_windows)
    background_amplitude_values = np.array(background_amplitude)
    background_amplitude_values = background_amplitude_values.reshape(
        window_nr, channel_number, 3
    )

    index = 0

    for w in range(len(windows)):

        if seizure_or_non_seizure_segment_indexes is not None:
            window_index = seizure_or_non_seizure_segment_indexes[w]
        else:
            window_index = w

        time = window_index * seconds
        index = find_background_amplitude_index(time, window_nr)

        for ch in range(channel_number):

            cA5, cD5, cD4, cD3, cD2, cD1 = dwt_coeffs(windows[w][ch])

            suma = len(cD3) + len(cD4) + len(cD5)
            size_D3 = len(cD3)
            size_D4 = len(cD4)
            size_D5 = len(cD5)

            absolute_energy_d3 = absolute_energy(cD3, size_D3, frequency)
            absolute_energy_d4 = absolute_energy(cD4, size_D4, frequency)
            absolute_energy_d5 = absolute_energy(cD5, size_D5, frequency)

            sum_absolute_energies = (
                absolute_energy_d3 + absolute_energy_d4 + absolute_energy_d5
            )
            relative_energy_d3 = relative_energy(
                absolute_energy_d3, sum_absolute_energies
            )
            relative_energy_d4 = relative_energy(
                absolute_energy_d4, sum_absolute_energies
            )
            relative_energy_d5 = relative_energy(
                absolute_energy_d5, sum_absolute_energies
            )

            absolute_amplitude_in_epoch_d3 = calculate_amplitudes(cD3)
            absolute_amplitude_in_epoch_d4 = calculate_amplitudes(cD4)
            absolute_amplitude_in_epoch_d5 = calculate_amplitudes(cD5)

            background_amplitude_d3 = background_amplitude_values[index][ch][0]
            background_amplitude_d4 = background_amplitude_values[index][ch][1]
            background_amplitude_d5 = background_amplitude_values[index][ch][2]

            relative_amplitude_d3 = (
                absolute_amplitude_in_epoch_d3 / background_amplitude_d3
            )
            relative_amplitude_d4 = (
                absolute_amplitude_in_epoch_d4 / background_amplitude_d4
            )
            relative_amplitude_d5 = (
                absolute_amplitude_in_epoch_d5 / background_amplitude_d5
            )

            coefficient_variation_d3 = coefficient_variation(cD3, size_D3)
            coefficient_variation_d4 = coefficient_variation(cD4, size_D4)
            coefficient_variation_d5 = coefficient_variation(cD5, size_D5)

            FI3 = fluctuation_index(cD3, size_D3)
            FI4 = fluctuation_index(cD4, size_D4)
            FI5 = fluctuation_index(cD5, size_D5)

            feature = [
                relative_energy_d3,
                relative_energy_d4,
                relative_energy_d5,
                relative_amplitude_d3,
                relative_amplitude_d4,
                relative_amplitude_d5,
                coefficient_variation_d3,
                coefficient_variation_d4,
                coefficient_variation_d5,
                FI3,
                FI4,
                FI5,
            ]
            features.append(feature)

    features = np.array(features)
    return features


""" 

Funkcija serializuoja duomenis ir juos išsugo į PKL failą
Parametrai:
x_test - testavimo x požymių vektorius
y_test - y klasės etikečių testavimo vektorius
x_train - traniravimo x požymių vektorius
y_train - y klasės etikečių treniravimo vektorius
channels_per_patient - paciento kanalų skaičius
all_seizures - priepuolių skaičius
patient_number - pacientų skaičius
windows_per_patient - paciento segmentų skaičius

"""


def save_features_to_pkl_file(
    x_test,
    y_test,
    x_train,
    y_train,
    channels_per_patient,
    all_seizures,
    patient_number,
    windows_per_patient,
):
    variables = {
        "XTest": x_test,
        "yTest": y_test,
        "XTrain": x_train,
        "yTrain": y_train,
        "channelNumber": channels_per_patient,
        "seizures": all_seizures,
        "patientsNumber": patient_number,
        "windowNumber": windows_per_patient,
    }

    # Save with pickle
    with open("featuresChanged.pkl", "wb") as f:
        pickle.dump(variables, f)
    print("Saved features to featuresChanged.pkl")
