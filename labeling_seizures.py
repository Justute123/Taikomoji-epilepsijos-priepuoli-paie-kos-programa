import numpy as np
import math

"""

Funkcija, kuri užpildo y vektorių vienetais, ten kur buvo įvykę priepuoliai.
Iš pradžių gaunami priepuolių intervalai, pagal kuriuos nustatomi segmentų indeksai, kuriuose įvyko priepuoliai.
Pvz. jeigu priepuolis įvyko 16.3 ir 17.8 indeksuose, tai priepuolis įvyko 16-18 segmentuose.
Parametrai:
window_number - langų skaičius
seizure_intervals - priepuolių intervalai
seconds - vieno segmento trukmė sekundėmis
Gražinamas:
y - klasės etikečių vektorius
seizures_number - priepuolių segmentų skaičius
nonseizures_number - nepriepuolių segmentų skaičius

"""


def fill_y_with_seizures(window_number, seizure_intervals, seconds):

    y = np.zeros(window_number)

    for start_seizure_time, end_seizure_time in seizure_intervals:

        start = int(math.floor(start_seizure_time / seconds))
        end = int(math.ceil(end_seizure_time / seconds))

        # end +1 tam, kad paimtu paskutini elementa
        y[start : (end + 1)] = 1

    seizures_number = np.sum(y == 1)
    nonseizures_number = np.sum(y == 0)

    return y, seizures_number, nonseizures_number


"""
Ši funkcija skaičiuoja priepuolių skaičių y vektoriuje ir veikia tokiu principu:
jeigu esamo elemento ir sekančio skirtumas yra -1, priepuolis įvyko.
Parametrai:
y - klasės vektorius
Grąžinama:
count - priepuolių skaičius
"""


def calculate_number_of_seizures(y):

    # išlygina y vektorių, pvz jeigu [[0,1,1]] tai paverčiama į [0,1,1]
    y = np.asarray(y).flatten()

    difference_list = np.diff(y)

    count = 0

    for diff in difference_list:

        if diff == -1:

            count = count + 1

    if y[-1] == 1:

        count = count + 1

    return count


"""

Funkcija, kuri nuskaito iš tekstinio failo tikrą priepuolių skaičių ir jį paverčia į int formatą
Parametrai:
index - indeksas eilutės, kurioje yra nurodytas priepuolių skaičius
lines - tekstinio failo eilučių turinio sąrašas
Grąžinama:
seizure_number_converted_to_int - priepuolių skaičius paverstas į int

"""


def collect_seizure_number_from_txt(index, lines):

    index = index + 1
    seizure_number = lines[index].split(":")[1]

    seizure_number_converted_to_int = int(seizure_number)

    return seizure_number_converted_to_int


"""

Funkcija, kuri iš tekstinio failo išsaugo tik reikiamus skaičius be teksto: priepuolio pradžios ir pabaigos laiką.
Parametrai:
index - indeksas eilutės, kurioje yra nurodytas priepuolių skaičius
lines - tekstinio failo eilučių turinio sąrašas
seizure_number_converted_to_int - priepuolių skaičius paverstas į int
Gražinamas:
seizure_intervals - priepuolių laikų sąrašas

"""


def calculate_seizure_intervals(seizure_number_converted_to_int, index, lines):

    seizure_intervals = []
    z_start = 1
    z_end = 2
    index = index + 1

    for i in range(seizure_number_converted_to_int):

        seizure_start = lines[index + (2 * i + z_start)].split(":")[1]

        seizure_start_final = int(seizure_start.split()[0])

        seizure_end = lines[index + (2 * i + z_end)].split(":")[1]

        seizure_end_final = int(seizure_end.split()[0])

        seizure_intervals.append((seizure_start_final, seizure_end_final))

    return seizure_intervals


"""
Funkcija, kuri nuskaito tekstinį failo ir ieško EDF failo pavadinimo ir nustato indeksą,
kurį panaudoja ieškant priepuolių skaičiui ir priepuolių laikų intervalams.
Parametrai:
txt_file_path - txt failo kelias
word - ieškomas žosdis faile (EDF failo pavadinimas)
Grąžinama:
seizure_intervals - priepuolių intervalai

"""


def formating_seizure_intervals(txt_file_path, word):

    seizure_intervals = []

    if txt_file_path:

        with open(txt_file_path[0]) as f:

            lines = f.readlines()

        index = -1

        for row in lines:
            if row.find(word) != -1:
                index = lines.index(row)

        if index == -1:
            print("ieškomas žodis nerastas")
            return False

        seizure_number_converted_to_int = collect_seizure_number_from_txt(index, lines)
        seizure_intervals = calculate_seizure_intervals(
            seizure_number_converted_to_int, index, lines
        )
        return seizure_intervals

    else:
        print("no txt file in folder")

    return seizure_intervals
