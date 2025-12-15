"""
Vykdomas EEG signalų išankstinis signalų filtravimas. Atliekamas siaurajuosčio slopinimo filtravimas ir juostinis filtravimas.

Parametrai:
raw - raw objektas
notch_freq - siaurajuosčio slopinimo filtravimo dažnis
band_low-freq - žemo dažnio riba
band_high_freq - aukšto dažnio riba

grąžinami:
signals - išfiltruoti signalai

"""


def preprocess_signals(raw, notch_freq, band_low_freq, band_high_freq):

    raw.notch_filter(freqs=notch_freq)
    raw.filter(l_freq=band_low_freq, h_freq=band_high_freq)

    signals = raw.get_data()

    return signals


"""

Vykdomas kanalų filtravimas. Atrenkami tik tie kanalai (treniravimui), kurie yra sąraše.
Parametrai:
channel_names - EDF faile naudojami kanalai
Grąžinama:
pick - tik kanalai, kurie yra iš sąrašo

"""


def pick_names(channel_names):

    ch_pick_names = [
        "FP1-F7",
        "F7-T7",
        "T7-P7",
        "P7-O1",
        "FP1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "FP2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "FP2-F8",
        "F8-T8",
        "T8-P8-0",
        "P8-O2",
        "FZ-CZ",
        "CZ-PZ",
        "P7-T7",
        "T7-FT9",
        "FT9-FT10",
        "FT10-T8",
        "T8-P8-1",
    ]

    pick = []

    for ch in ch_pick_names:
        if ch in channel_names:
            pick.append(ch)
    return pick
