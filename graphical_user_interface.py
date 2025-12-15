import mne

mne.viz.set_browser_backend("qt")

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import datetime

import numpy as np
import ntpath
import sys
from pyedflib import highlevel
import os
import joblib

from patients_info import pick_names
from preprocessing import preprocess_signals
from segmentation import split_to_windows
from feature_extraction import feature_calculation
from labeling_seizures import fill_y_with_seizures, formating_seizure_intervals
from clasification import calculate_metrics, testing, validation
from utils import plot_EEG_graph
from sklearn.metrics import confusion_matrix

from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
    QMainWindow,
    QApplication,
    QPushButton,
    QWidget,
    QTabWidget,
    QLabel,
)


class GUI(QMainWindow):
    """

    Inicializuoja klasės objektą ir sukuria pagrindinį programos langą su nurodytomis ypatybėmis

    """

    def __init__(self):
        super().__init__()
        self.isFileUploaded = False
        self.is_preprocessed = False
        self.is_channeled = False
        self.is_searched = False
        self.is_preprocsed_fields_created = False
        self.preprocessed_signals = None
        self.choosed_channels = None
        self.file_path = None
        self.file_name_in_text = None
        self.lastWindow = None
        self.lastPredicted = None
        self.seizures_number = None
        self.seizures_times = None
        self.channel_names = None
        self.raw_seg = None
        self.seconds = 4
        self.filtered_segments = []

        self.setWindowTitle("Epilepsijos priepuolių programa")
        self.setFont(QFont("Arial", 15))
        self.resize(1000, 700)
        tab_widget = self.create_tabs()
        self.setCentralWidget(tab_widget)

    """
    
    Sukuria tris atskirus skirtukus (angl. tabs), kuriuose yra pridedami UI elementai, kurie yra sugeneruojami first_layout(), second_layout() ir third_layout() funkcijose 
    
    """

    def create_tabs(self):

        tabs = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()

        self.layout1 = QGridLayout()
        self.layout1.setAlignment(Qt.AlignTop)
        self.layout1.setContentsMargins(10, 40, 0, 0)
        self.file_name_label, self.button_choose_file, self.patient_info_table = (
            self.first_layout()
        )

        tab1.setLayout(self.layout1)

        self.layout2 = QGridLayout()
        self.layout2.setContentsMargins(20, 30, 0, 0)
        self.layout2.setAlignment(Qt.AlignTop)
        self.info2_label, self.right_side_layout_2 = self.second_layout(self.layout2)
        tab2.setLayout(self.layout2)

        self.layout3 = QGridLayout()
        self.layout3.setContentsMargins(20, 30, 0, 0)
        self.layout3.setAlignment(Qt.AlignTop)
        self.info3_label, self.right_side_layout_3 = self.third_layout(self.layout3)
        tab3.setLayout(self.layout3)

        self.button_choose_file.clicked.connect(self.execute)

        tabs.addTab(tab1, "EEG įrašo informacija")
        tabs.addTab(tab2, "Išankstinis signalų apdorojimas")
        tabs.addTab(tab3, "Priepuolių paieška")

        return tabs

    """ 
    
    Sukuria pirmojo skirtuko grafinius elementus 
    
    grąžina: failo pavadinimo eteiketę, failo pasirinkimo mygtuką ir paciento duomenų lentelę
    
    """

    def first_layout(self):
        self.file_name_label = QLabel()
        self.file_name_label.setText("Failas nepasirinktas")
        self.file_name_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.layout1.addWidget(self.file_name_label, 0, 0)
        self.layout1.setSpacing(30)

        self.button_choose_file = QPushButton("Įkelkite failą EDF formatu")
        self.button_choose_file.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.button_choose_file.setFixedSize(300, 50)
        self.layout1.addWidget(self.button_choose_file, 1, 0)

        self.patient_info_table = QTableWidget()
        self.patient_info_table.setRowCount(7)
        self.patient_info_table.setColumnCount(2)
        self.patient_info_table.hide()
        self.patient_info_table.setStyleSheet("font-size: 20px;")
        self.layout1.addWidget(self.patient_info_table, 2, 0, 1, 2)

        self.patient_info_table.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.patient_info_table.horizontalHeader().setStretchLastSection(False)
        self.patient_info_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

        return self.file_name_label, self.button_choose_file, self.patient_info_table

    """ 
    
    Sukuria antrojo skirtuko grafinius elementus 
        
    grąžina: pranešimo etiketę, dešinės pusės išdėstymą
        
    """

    def second_layout(self, layout2):

        self.right_side_layout_2 = QGridLayout()
        layout2.addLayout(self.right_side_layout_2, 1, 5)
        self.info2_label = QLabel()
        self.info2_label.setText(
            "Prieš atlikdami kanalų filtravimą pasirinkite failą pirmoje dalyje"
        )
        layout2.addWidget(self.info2_label, 0, 0)

        layout2.setColumnStretch(0, 1)
        layout2.setColumnStretch(1, 2)
        self.right_side_layout_2.setAlignment(Qt.AlignTop)

        return self.info2_label, self.right_side_layout_2

    """ 
    
    Sukuria trečiojo skirtuko grafinius elementus 
        
    grąžina: pranešimo etiketę, dešinės pusės išdėstymą
        
    """

    def third_layout(self, layout3):

        self.right_side_layout_3 = QGridLayout()
        self.layout3.addLayout(self.right_side_layout_3, 1, 2)

        self.info3_label = QLabel()
        self.info3_label.setText(
            "Prieš priepuolių paiešką būtina atlikti išankstinį signalų apdorojimą ir kanalų pasirinkimus"
        )
        layout3.addWidget(self.info3_label, 0, 0)

        layout3.setColumnStretch(0, 3)
        layout3.setColumnStretch(1, 1)
        layout3.setColumnStretch(2, 1)

        return self.info3_label, self.right_side_layout_3

    """
    Vykdančioji funkcija, kuri atlieka paciento informacijos išvedimą ir, 
    jeigu failas yra įkeltas atliekamas kanalų filtravimas ir signalų išankstinis apdorojimas
    
    """

    def execute(self):

        self.show_patient_info(self.file_name_label, self.patient_info_table)

        if hasattr(self, "channel_message"):
            self.channel_message.setText("")

        if hasattr(self, "preprocess_message"):
            self.preprocess_message.setText("")

        if self.isFileUploaded == True:

            self.info2_label.hide()
            self.select_channels_from_list(self.channel_names, self.layout2)
            self.channel_action()

            self.preprocess_action()

    """ 
    Failo įkėlimas
    grąžinama: failo pavadinimas ir failo kelias
    """

    def upload_file(self):

        file_name_label = QLabel()
        filepath = QFileDialog.getOpenFileName(None, "Įkėlimas", ".", "*.edf")
        filename = os.path.basename(filepath[0])
        file_name_label.setText(filename)

        self.file_path = filepath[0]
        self.file_name_in_text = filename

        return filename, filepath

    """
    
    Išvedama EDF failo informacija ir ji atvaizuojama lentelės pavidale
    Parametrai: failo pavadinimo etiketė ir paciento lentelės elementas
    
    """

    def show_patient_info(self, file_name_label, patient_info_table):

        filename, filepath = self.upload_file()

        if filename:
            file_name_label.setText(f"Pasirinktas failas: {filename}")
            self.isFileUploaded = True

            signals, signal_headers, header = highlevel.read_edf(filepath[0])
            raw = mne.io.read_raw_edf(filepath[0], preload=True, infer_types=True)
            raw.set_meas_date(None)
            self.channel_names = raw.ch_names

            patientname = header.get("patientname")
            patientcode = header.get("patientcode")
            patientgender = header.get("gender")
            birthdate = header.get("birthdate")
            startdate = header.get("startdate")
            frequency = raw.info["sfreq"]
            number_channels = raw.info["nchan"]

            patient_info_table.setHorizontalHeaderLabels(
                ["Pavadinimas", "Apibūdinimas"]
            )
            patient_info_table.setItem(0, 0, QTableWidgetItem("Paciento vardas"))
            patient_info_table.setItem(0, 1, QTableWidgetItem(str(patientname)))
            patient_info_table.setItem(1, 0, QTableWidgetItem("Paciento kodas"))
            patient_info_table.setItem(1, 1, QTableWidgetItem(str(patientcode)))
            patient_info_table.setItem(2, 0, QTableWidgetItem("Lytis"))
            patient_info_table.setItem(2, 1, QTableWidgetItem(str(patientgender)))
            patient_info_table.setItem(3, 0, QTableWidgetItem("Gimimo data"))
            patient_info_table.setItem(3, 1, QTableWidgetItem(str(birthdate)))
            patient_info_table.setItem(4, 0, QTableWidgetItem("Įrašymo data"))
            patient_info_table.setItem(4, 1, QTableWidgetItem(str(startdate)))
            patient_info_table.setItem(5, 0, QTableWidgetItem("Dažnis"))
            patient_info_table.setItem(5, 1, QTableWidgetItem(str(frequency) + " Hz."))
            patient_info_table.setItem(6, 0, QTableWidgetItem("Kanalų skaičius"))
            patient_info_table.setItem(6, 1, QTableWidgetItem(str(number_channels)))
            patient_info_table.show()

    # jeigu naudotojas paspaudzia pazymeti visus, tai pazyimimi visi kanalai, kai pavieniai tai nereikia logikos nes pazymi pats zmogus ir tereikia patikrinti busena
    """
    Funkcija atlieka visų kanalų pažymėjimą, kai naudotojas paspaudžia ant pažymėti visus kanalus
    
    Parametrai:
    item yra sąrašo elementas, kurį naudotojas paspaudė
    channel_list yra kanalų sąrašas, kur kiekvienas kanalas turi langelį, kurį galima pažymėti arba atžymėti
    
    """

    def mark_all_channels(self, item, channel_list):

        if item.text() == "Pažymėti visus kanalus":
            n = channel_list.count()
            for i in range(1, n):
                ch = channel_list.item(i).setCheckState(item.checkState())

    """
    
    Funkcija, kuri sukuria kanalų sąrašą su galimybe pažymėti tiek visus kanalus vienu metu, tiek individualiai po vieną ir prideda į antrą skirtuką
    
    Parametrai:
    channel_names yra kanalų pavadinimai
    layout yra išdėstymo elementas, kuriame ir bus atvaizuotas kanalų sąrašas
    
    """

    def select_channels_from_list(self, channel_names, layout):

        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.channel_list.setStyleSheet("font-size: 18px;")
        self.channel_list.setFixedSize(400, 500)  # width, height

        # iterpiamas elementas pazymeti
        select_all_channels = QListWidgetItem("Pažymėti visus kanalus")
        select_all_channels.setFlags(
            select_all_channels.flags() | QtCore.Qt.ItemIsUserCheckable
        )
        select_all_channels.setCheckState(QtCore.Qt.Unchecked)
        self.channel_list.addItem(select_all_channels)

        for ch in channel_names:
            item = QListWidgetItem(ch)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.channel_list.addItem(item)

        # vykdys jeigu pasirinkti visi
        self.channel_list.itemChanged.connect(
            lambda item: self.mark_all_channels(item, self.channel_list)
        )
        layout.addWidget(self.channel_list, 1, 0)

    """
    
    Sukuriama kanalų pasirinkimo pranešimo elementas. Visų pirma patikrina ar toks elementas jau egzistuoja, jeigu taip - tai jį išvalo, jeigu ne - sukuria naują
    Gražina:
    channel_message yra kanalų pranešimo etiketė
    
    """

    def create_channel_message(self):

        if hasattr(self, "channel_message"):
            self.channel_message.setText("")
        else:
            self.channel_message = QLabel()
            self.layout2.addWidget(self.channel_message, 4, 0)
            self.channel_message.setAlignment(Qt.AlignCenter)
        return self.channel_message

    """
    
    Nustato pasirinktus kanalus, patikrina ar kanalų žinutės etiketė buvo sukurta, pakeičia būseną, kad kanalai buvo pažymėti ir išveda atitinkamą pranešimą
    
    
    """

    def pick_channels(self):

        n = self.channel_list.count()

        # check what channels were picked
        picked_channels = []
        for i in range(1, n):
            item = self.channel_list.item(i)
            if item.checkState() == Qt.Checked:
                picked_channels.append(item.text())

        # if label was created chedck
        channel_message = self.create_channel_message()
        if len(picked_channels) > 0:
            self.is_channeled = True
            self.ready_for_seizure_detection()
            self.choosed_channels = picked_channels
            channel_message.setText("Sėkmingai pasirinkti kanalai")
            channel_message.setStyleSheet("color: green;font-size: 18px;")
        else:
            channel_message.setText("Pasirinkite bent vieną kanalą")
            channel_message.setStyleSheet("color: red;font-size: 18px;")

    """
   
    Sukuriamas kanalų pasirinkimo mygtukas ir prijungiamas prie funkcijos, kuris nustato pasirinktus kanalus
   
    """

    def channel_action(self):

        self.layout2.filter_button = QPushButton("Atlikti kanalų filtravimą")
        self.layout2.addWidget(self.layout2.filter_button, 3, 0)

        self.layout2.filter_button.clicked.connect(lambda: self.pick_channels())

    """
    
    Funkcijoje yra sukuriamos išankstinio apdorojimo pavadinimų etiketės, įvesties laukai skirti naudotojui, mygtukas, kuris skirtas atlikti išankstinį signalų apdorjimą
    Nustatoma būsena, kad išankstinių signalų apdorojimo grafiniai elementai buvo sukurti.
    
    """

    def preprocess_fields(self):

        self.preprocess_message = QLabel()
        # sulauzo teksta
        self.preprocess_message.setWordWrap(True)

        labeltext1 = QLabel()
        labeltext1.setText("Siaurajuostis slopinimo filtras: ")
        self.right_side_layout_2.addWidget(labeltext1, 0, 0)
        self.textbox1 = QLineEdit()
        self.textbox1.setPlaceholderText("Įveskite reikšmę")
        self.right_side_layout_2.addWidget(self.textbox1, 0, 1)

        labeltext2 = QLabel()
        labeltext2.setText("Žemo dažnio filtro riba: ")
        self.right_side_layout_2.addWidget(labeltext2, 1, 0)
        self.textbox2 = QLineEdit()
        self.textbox2.setPlaceholderText("Įveskite reikšmę")
        self.right_side_layout_2.addWidget(self.textbox2, 1, 1)

        labeltext3 = QLabel()
        labeltext3.setText("Aukšto dažnio filtro riba: ")
        self.right_side_layout_2.addWidget(labeltext3, 2, 0)
        self.textbox3 = QLineEdit()
        self.textbox3.setPlaceholderText("Įveskite reikšmę")
        self.right_side_layout_2.addWidget(self.textbox3, 2, 1)

        self.preprocess = QPushButton("Apdoroti signalus")
        self.right_side_layout_2.addWidget(self.preprocess, 3, 1)

        self.right_side_layout_2.addWidget(self.preprocess_message, 6, 0, 1, 2)

        self.is_preprocsed_fields_created = True

    """
    
    Pagalbinė funkcija, kuri tikrina ar skaičius gali būti konvertuojamas į dešimtainį formatą. Jeigu naudotojo įvestas skaičius nesikonvertuoja išvedamas klaidos pranešimas.
    Parametrai:
    filter_value yra naudotojo įvestas skaičius
    
    """

    def is_number_float(self, filter_value):

        try:

            float(filter_value)
            return True

        except ValueError:

            self.preprocess_message.setText("Įrašykite skaitines reikšmes")
            self.preprocess_message.setStyleSheet("color: red;font-size: 18px;")
            return False

    """
    
    Pagalbinė funkcija, skirta raudonai klaidos žinutei išvesti.
    Parametras:
    message yra žinutė, kurią norima išvesti
    
    """

    def print_validation_message(self, message):

        self.preprocess_message.setText(message)
        self.preprocess_message.setStyleSheet("color: red;font-size: 18px;")

    """
    
    Funkcija, skirta naudotojo įvesčiai tikrinti. Jeigu vertės tinka, tada atliekamas išankstinis signalų apdorojimas, nustatoma būsena, kad šis procesas buvo įvykdytas ir išvedamas sėkmės pranešimas
    grąžinama:
    signals - tai signalai, kuriems buvo pritaikytas išankstinis signalų apdorojimas
    
    """

    def preprocess_field_validation(self, raw):

        notch = self.textbox1.text()
        band_low = self.textbox2.text()
        band_high = self.textbox3.text()

        if notch == "" or band_low == "" or band_high == "":

            self.print_validation_message("Įrašykite reikšmes")

            return

        if (
            self.is_number_float(notch) == True
            and self.is_number_float(band_low) == True
            and self.is_number_float(band_high) == True
        ):

            notch_to_float = float(notch)
            band_low_to_float = float(band_low)
            band_high_to_float = float(band_high)

            if notch_to_float < 0 or notch_to_float > 60:

                self.print_validation_message(
                    "Siaurajuosčio slopinimo filtro dažnis turi būti mažiau arba lygu 60"
                )

                return

            if band_low_to_float < 0.1 or band_low_to_float > 1:

                self.print_validation_message(
                    "Žemo dažnio filtravimo riba turi būti 0.1-1 diapazone"
                )

                return
            if band_high_to_float < 30 or band_high_to_float > 120:

                self.print_validation_message(
                    "Aukšto dažnio filtravimo riba turi būti 30-120 diapazone"
                )

                return
            else:

                preprocess_signals(
                    raw, notch_to_float, band_low_to_float, band_high_to_float
                )
                self.is_preprocessed = True
                self.ready_for_seizure_detection()
                self.preprocess_message.setText(
                    "Išankstinis signalų apdorojimas sėkmingai pavyko"
                )
                self.preprocess_message.setStyleSheet("color: green;font-size: 18px;")
                self.preprocessed_signals = raw
                signals = raw.get_data()
                return signals

    """
    
    Vykdomas išankstinis signalų filtravimas
    
    """

    def preprocess_action(self):

        raw = mne.io.read_raw_edf(self.file_path, preload=True, infer_types=True)

        if self.is_preprocsed_fields_created == False:
            self.preprocess_fields()

        self.preprocess.clicked.connect(lambda: self.preprocess_field_validation(raw))

    """
    
    Tikrinama, ar yra pasiruošta priepuolių paieškai, ar pasirinkti norimi kanalai ir, ar buvo atliktas išankstinis apdorojimas
    
    """

    def ready_for_seizure_detection(self):

        if self.is_channeled == True and self.is_preprocessed == True:
            self.epilepsy_detection_action()

    """
    
    Trečiame skirtuke pridedamas priepuolių paieškos mygtukas ir 
    jeigu paieška buvo atlikta atvaizduojami SVM validacijos eksporto mygtukas bei paieškos rezultatų eksporto mygtukas
    
    """

    def epilepsy_detection_action(self):

        self.info3_label.hide()

        if not hasattr(self, "seizures_search_button"):

            self.seizures_search_button = QPushButton("Vykdyti priepuolių paiešką")
            self.seizures_search_button.setSizePolicy(
                QSizePolicy.Fixed, QSizePolicy.Fixed
            )
            self.seizures_search_button.setFixedSize(300, 50)
            self.layout3.addWidget(self.seizures_search_button, 1, 0)
            self.seizures_search_button.clicked.connect(lambda: self.run())
        else:
            self.seizures_search_button.show()

        if not hasattr(self, "verification_button"):

            self.verification_button = QPushButton("SVM validacijos eksportas")
            self.verification_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.verification_button.setFixedSize(300, 50)
            self.right_side_layout_3.addWidget(self.verification_button, 1, 0)
            self.verification_button.clicked.connect(lambda: self.verification())
            print(self.is_searched)
            if self.is_searched == False:

                self.verification_button.hide()
            else:
                self.verification_button.show()

        if not hasattr(self, "export_button"):

            self.export_button = QPushButton("Paieškos rezultatų eksportas")
            self.export_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.export_button.setFixedSize(300, 50)
            self.right_side_layout_3.addWidget(self.export_button, 2, 0)
            self.export_button.clicked.connect(lambda: self.exportas())

            if self.is_searched == False:

                self.export_button.hide()
            else:
                self.export_button.show()

    """
    
    Sukuriamas progreso juostos elementas, atliekama priepuolių paieška, nustatoma būsena, kad paieška buvo įvykdyta,
    išvedamas rastų priepuolių skaičius bei priepuolių segmentai, kuriuose nurodoma pardžia ir pabaiga. Šie segmentai atvaizduojami sąrašp pavidalu.,
    suteikiama galimybė, kad sąrašo elementas būtų paspaudžiamas, norint atvaizduoti EEG segmento grafiką
    
    """

    def run(self):

        pbar = QProgressBar()
        self.layout3.addWidget(pbar, 2, 0, 1, 3)
        pbar.show()

        QApplication.processEvents()
        times, indexes, number, raw, window_number, y_predicted_after_reshaping = (
            self.testing_part(pbar)
        )

        self.seizures_number = number
        self.seizures_times = times
        self.raw_seg = raw

        self.verification_button.show()
        self.export_button.show()
        self.is_searched = True

        if hasattr(self, "seizure_number_label"):
            self.seizure_number_label.hide()

        self.seizure_number_label = QLabel()
        self.layout3.addWidget(self.seizure_number_label, 3, 0)

        seizures_list = QListWidget()
        seizures_list.setFixedSize(400, 500)
        self.layout3.addWidget(seizures_list, 4, 0)

        i = 1

        for start, end in times:

            if end <= start:
                continue

            self.filtered_segments.append((start, end))
            seizures_list.addItem(f"{i} nr. {start} s – {end} s")
            i = i + 1

        self.seizure_number_label.setText(
            f"Priepuolių skaičius: {len(self.filtered_segments)}"
        )

        seizures_list.itemClicked.connect(lambda item: self.segment_click(item))

    """
    
    Funkcija, kuri gauna pasirinkto segmento sąrašo indeksą ir išsaugo segmento pradžios ir pabaigos laiką,
    juos konvertuoja ir į speocialų indeksą signalo duomenų sąraše, taip gaunamos signalo amplitudžių vertės ir laikai,
    kurie bus naudojami EEG grafikui piešti
    
    Parametrai:
    item - sąrašo elemento indeksas
    
    """

    def segment_click(self, item):

        selected_list = item.listWidget()
        selected_segment_index = selected_list.row(item)

        start_time = self.filtered_segments[selected_segment_index][0]
        end_time = self.filtered_segments[selected_segment_index][1]
        start_index, end_index = self.raw_seg.time_as_index([start_time, end_time])
        if end_index <= start_index:
            return
        segment_data, segment_times = self.raw_seg[:, start_index:end_index]
        print("segment_data shape:", np.shape(segment_data))
        print("segment_times length:", len(segment_times))
        print("choosed_channels:", self.choosed_channels)
        print("start_idx:", start_index, "end_idx:", end_index)
        print("signal length:", segment_data.shape[1])

        plot_EEG_graph(segment_data, segment_times, self.choosed_channels)

    """
    Šioje funckijoje yra pasirenkami jau signalai, kuriems buvo pritaikytas išankstinis signalų apdorojimas, jie yra suskirstomi į segmentus ir iš jų išskyriami požymiai.
    Po to su išsaugoto SVM modelio pagalba yra prognozuojami priepuolių segmentai ir jų laikai bei segmentų skaičius
    
    grąžinama:
    times - priepuolių laikai
    indexes - priepuolių indeksai
    number - priepuolių skaičius
    preprocessed_signals - signalai, kuriems pritaikytas išankstinis signalų apdprojimas
    window_number - segmentų skaičius
    y_predicted_after_reshaping - y klasės etikečių vektorius, prognozuotiems priepuoliams 
    """

    def testing_part(self, pbar=None):

        if pbar is not None:
            pbar.setMinimum(0)
            QApplication.processEvents()
            pbar.setMaximum(100)

        XTest = []

        frequency = self.preprocessed_signals.info["sfreq"]
        sample_size = self.preprocessed_signals.n_times
        one_window_signal_number = self.seconds * int(frequency)
        duration_all_signal = sample_size / frequency
        window_number = int(sample_size / one_window_signal_number)
        channel_number = len(self.choosed_channels)

        signals = self.preprocessed_signals.get_data()

        window = split_to_windows(
            signals, channel_number, window_number, one_window_signal_number
        )
        if pbar is not None:
            pbar.setValue(10)
            QApplication.processEvents()

        # calculate all features in all segments
        all_feature = feature_calculation(
            signals,
            window,
            frequency,
            channel_number,
            self.seconds,
            duration_all_signal,
        )

        XTest.append(all_feature)

        if pbar is not None:
            pbar.setValue(50)
            QApplication.processEvents()

        svc_clf = joblib.load("seizureFile.sav")
        scaler = joblib.load("scalerFile.sav")

        y_predicted_after_reshaping = testing(
            XTest, window_number, channel_number, scaler, svc_clf, None
        )

        self.lastWindow = window_number
        self.lastPredicted = y_predicted_after_reshaping

        if pbar is not None:
            pbar.setValue(70)
            QApplication.processEvents()

        indexes = []
        times = []
        start = None
        end = None

        for i in range(len(y_predicted_after_reshaping)):

            # detect seizure start
            if y_predicted_after_reshaping[i] == 1 and (
                i == 0 or y_predicted_after_reshaping[i - 1] == 0
            ):

                start = i

            # detect seizure end
            if y_predicted_after_reshaping[i] == 1 and (
                i == len(y_predicted_after_reshaping) - 1
                or y_predicted_after_reshaping[i + 1] == 0
            ):

                end = i

                indexes.append((start, end))
                times.append((start * self.seconds, end * self.seconds))

        # priepuoliu skaicius
        number = len(indexes)

        if pbar is not None:
            pbar.setValue(100)
            QApplication.processEvents()

        return (
            times,
            indexes,
            number,
            self.preprocessed_signals,
            window_number,
            y_predicted_after_reshaping,
        )

    """
    Tikrų priepuolių intervalai, kurie gauti nuskaičius naudotojo įkeltą tekstinį failą. Funkcija naudojama SVM modelio validacijai.
    
    """

    def real_seizure_intervals_to_pdf(
        self, c, current_time, seizure_intervals, x_coord, y_coord
    ):

        c.drawString(
            50,
            760,
            f"Data: {current_time}, failo pavadinimas: {self.file_name_in_text}",
        )
        c.drawString(50, 740, f"Tikrų priepuolių laikai: ")

        count = 1

        for seizure in seizure_intervals:

            c.drawString(
                x_coord,
                y_coord,
                f"nr: {count}. priepuolio pradžia: {seizure[0]} sec., priepuolio pabaiga: {seizure[1]} sec.",
            )

            y_coord = y_coord - 30
            count = count + 1
        return y_coord

    """
    Funkcija, kuri išveda kiek priepuolių ir nepriepuoliųsegmentų buvo teisingai nustatyta, o kiek klaidingai.
    
    """

    def segments_statistics_to_pdf(self, seizure_intervals, c, y_coord):
        y_test_per_patient = []
        y, ones, zeros = fill_y_with_seizures(
            self.lastWindow, seizure_intervals, seconds=self.seconds
        )

        y_test_per_patient.append(y)

        cmTest = confusion_matrix(
            y[: len(self.lastPredicted)],
            self.lastPredicted,
            labels=[0, 1],
        )

        TN = cmTest[0, 0]
        FP = cmTest[0, 1]
        FN = cmTest[1, 0]
        TP = cmTest[1, 1]
        y_coord = y_coord - 40
        c.drawString(50, y_coord, f"Tikrų priepuolių segmentai: ")
        y_coord = y_coord - 20
        c.drawString(50, y_coord, f"Teisingai neigiami priepuolių segmentai: {TN}")
        y_coord = y_coord - 20
        c.drawString(50, y_coord, f"Klaidingai teigiami priepuolių segmentai: {FP}")
        y_coord = y_coord - 20
        c.drawString(50, y_coord, f"Klaidingai neigiami priepuolių segmentai: {FN}")
        y_coord = y_coord - 20
        c.drawString(50, y_coord, f"Teisingai teigiami priepuolių segmentai: {TP}")

        return y_coord, y_test_per_patient, TN, FP, FN, TP

    """
    Funkcija, kuri išveda SVM modelio metrikų rezultatus.
    
    """

    def metrics_to_pdf(self, TP, FP, FN, TN, c, y_coord):

        sensitivity, specificity, accurancy, precision, Fscore, fpr = calculate_metrics(
            TP, FP, FN, TN
        )

        ac = round(accurancy * 100, 2)
        sens = round(sensitivity * 100, 2)
        spec = round(specificity * 100, 2)
        prec = round(precision * 100, 2)
        f1 = round(Fscore * 100, 2)

        y_coord = y_coord - 40
        c.drawString(50, y_coord, f"Metrikos: ")
        y_coord = y_coord - 20
        c.drawString(50, y_coord, f"Testo tikslumas: {ac} %")
        y_coord = y_coord - 20
        c.drawString(50, y_coord, f"Testo jautrumas: {sens} %")
        y_coord = y_coord - 20
        c.drawString(50, y_coord, f"Testo specifiškumas: {spec} %")
        y_coord = y_coord - 20
        c.drawString(50, y_coord, f"Testo preciziškumas: {prec} %")
        y_coord = y_coord - 20
        c.drawString(50, y_coord, f"Testo F1: {f1} %")

    """
    Funkcija, kuri išveda SVM modelio validacijos rezulatus.
    
    """

    def verification(self):

        pdfmetrics.registerFont(
            TTFont(
                "LiberationSerif",
                "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            )
        )

        current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        message_box = QMessageBox()

        filepath = QFileDialog.getOpenFileName(None, "Įkėlimas", ".", "*.txt")
        c = canvas.Canvas("validacija.pdf", pagesize=A4)
        c.setFont("LiberationSerif", 12)
        width, height = A4
        center_x = width / 2
        y_coord = 700
        x_coord = 50

        c.drawCentredString(center_x, 800, "SVM modelio validacijos rezultatai")

        seizure_intervals = formating_seizure_intervals(
            filepath, ntpath.basename(self.file_name_in_text)
        )

        if seizure_intervals == False:

            message_box.setIcon(QMessageBox.Critical)
            message_box.setText(
                "Įkelkite teisingą tekstinį failą, kuriame yra reikiamas EDF failo pavadinimas"
            )
            message_box.setWindowTitle("Error")
            message_box.exec_()

            return
        else:

            y_coord = self.real_seizure_intervals_to_pdf(
                c, current_time, seizure_intervals, x_coord, y_coord
            )

            y_coord, y_test_per_patient, TN, FP, FN, TP = (
                self.segments_statistics_to_pdf(seizure_intervals, c, y_coord)
            )

            self.metrics_to_pdf(TP, FP, FN, TN, c, y_coord)

            message_box.setIcon(QMessageBox.Information)
            message_box.setText(
                "Validacijos rezultatai buvo sėkmingai išsaugoti į PDF failą validacija.pdf "
            )
            message_box.setWindowTitle("Information MessageBox")
            message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            retval = message_box.exec_()

            c.save()

    """
    
    Funkcija, kuri išveda rastų priepuolių segmentus ir jų skaičių į PDF failą
    
    """

    def exportas(self):

        pdfmetrics.registerFont(
            TTFont(
                "LiberationSerif",
                "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            )
        )

        current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        message = QMessageBox()
        c = canvas.Canvas("rezultatai.pdf", pagesize=A4)
        c.setFont("LiberationSerif", 12)

        width, height = A4
        center_x = width / 2

        c.drawCentredString(center_x, 800, "Epilepsijos priepuolių analizės rezultatai")
        c.drawString(
            50,
            760,
            f"Data: {current_time}, failo pavadinimas: {self.file_name_in_text}",
        )
        c.drawString(50, 740, f"Prognozuojamų priepuolių laikai: ")

        count = 1
        y_coord = 700
        x = 50

        for seizure in self.filtered_segments:

            c.drawString(
                x,
                y_coord,
                f"nr: {count}. priepuolio pradžia: {seizure[0]} sec., priepuolio pabaiga: {seizure[1]} sec.",
            )

            y_coord = y_coord - 30
            count = count + 1

        c.drawString(
            50, y_coord, f"Rastas priepuolių kiekis: {len(self.filtered_segments)}"
        )

        message.setIcon(QMessageBox.Information)
        message.setText(
            "Epilepsijos priepuolių analizės rezultatai buvo išsaugoti į PDF failą rezultatai.pdf "
        )
        message.setWindowTitle("Information MessageBox")
        message.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        retval = message.exec_()

        c.save()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    interface = GUI()
    interface.show()
    sys.exit(app.exec_())
