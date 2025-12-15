"""

Vykdomas segmentavimas. Padalijamas signalas į kanalų-langų atkarpas.

Parametrai:
signals - preprocesinti signalai
channel_number - kanalų skaičius
window_number - langų skaičius
one_window_signals_number - signalų skaičius viename lange
Grąžinama:
window - langų-kanalų signalo atkarpa


"""


def split_to_windows(signals, channel_number, window_number, one_window_signals_number):

    window = [[None for _ in range(channel_number)] for _ in range(window_number)]

    for i in range(window_number):

        start = i * one_window_signals_number
        end = (i + 1) * one_window_signals_number

        for j in range(channel_number):

            window[i][j] = signals[j, start:end]

    return window
