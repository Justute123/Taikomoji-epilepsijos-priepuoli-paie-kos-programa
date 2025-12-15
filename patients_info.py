import mne
from preprocessing import pick_names

"""

Funkcijoje yra gaunama paciento informacija iš EDF failo: kanalų skaičius, mėginių surinkimo dažnis. 
Apskaičiuojamas mėginių kiekis, vieno lango signalų kiekis, viso signalo trukmė, langų skaičius ir kanalų kiekis.

Parametrai:
file_path nusako EDF failo kelią
seconds nusako kiek sekundžių užtruks vienas segmentas

Grąžinama:
raw - raw objektas, kuris saugo EDF failo info
frquency - mėginių surinkimo dažnis
sample_size - mėginių kiekis (signalų duomenų taškų)
one_window_signals_number - duomenų taškai viename lange
window_number - langų skaičius
channel_number - kanalų skaičius

"""


def get_patient_details(file_path, seconds):

    raw = mne.io.read_raw_edf(file_path, preload=True, infer_types=True)
    raw.set_meas_date(None)
    frequency = raw.info["sfreq"]

    channel_names = raw.ch_names
    pick = pick_names(channel_names)
    raw.pick_channels(pick)

    sample_size = raw.n_times
    one_window_signals_number = seconds * int(frequency)
    duration_all_signal = sample_size / frequency
    window_number = int(sample_size / one_window_signals_number)
    channel_number = len(raw.ch_names)

    return (
        raw,
        frequency,
        sample_size,
        one_window_signals_number,
        duration_all_signal,
        window_number,
        channel_number,
    )
