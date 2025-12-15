from graphviz import Digraph

g = Digraph("Procesų seka", format="png")

g.attr(
    label="EEG signalų apdorojimo ir klasifikavimo procesų seka",
    labelloc="t",  # t = viršuje (top)
    fontsize="30",
)

# Mazgai
g.node("db", "CHB-MIT \n galvos odos \n duomenų bazė", shape="cylinder")
g.node("raw", "Neapdoroti \n CHB-MIT \n duomenys", shape="parallelogram")
g.node("det", "EDF failo informacijos išgavimas", shape="box")
g.node("pre", "Išankstinis signalų apdorojimas", shape="box")
g.node("seg", "Segmentavimas", shape="box")
g.node("label", "Etikečių klasės vektoriaus žymėjimas", shape="box")
g.node("bal", "Balansavimas", shape="box")
g.node("dwt", "DWT koeficientų apskaičiavimas", shape="box")
g.node("fet", "Požymių išskyrimas", shape="box")
g.node("clasific", "Klasifikavimas", shape="box")
g.node("post", "Galutinis rezultatų apdorojimas", shape="box")

g.node("train", "Treniravimui skirti duomenys\n(70%)", shape="box")
g.node("test", "Testavimui skirti duomenys\n(30%)", shape="box")

g.node("ml", "SVM mašininio mokymosi modelis", shape="box")

g.node("ac", "SVM tikslumas", shape="ellipse")
g.node("sens", "SVM jautrumas", shape="ellipse")
g.node("spre", "SVM specifiškumas", shape="ellipse")
g.node("prec", "SVM preciziškumas", shape="ellipse")
g.node("F1", "SVM F1 metrika", shape="ellipse")

# Briaunos
g.edge("db", "raw")
g.edge("raw", "det")
g.edge("det", "pre")
g.edge("pre", "seg")
g.edge("seg", "label")
g.edge("label", "bal")
g.edge("bal", "dwt")
g.edge("dwt", "fet")
g.edge("fet", "clasific")

g.edge("clasific", "train", label="70%")
g.edge("clasific", "test", label="30%")

g.edge("train", "post")
g.edge("test", "post")
g.edge("post", "ml")

g.edge("ml", "ac")
g.edge("ml", "sens")
g.edge("ml", "spre")
g.edge("ml", "prec")
g.edge("ml", "F1")


# Rezultatai
g.render("pipeline.png", cleanup=True)
