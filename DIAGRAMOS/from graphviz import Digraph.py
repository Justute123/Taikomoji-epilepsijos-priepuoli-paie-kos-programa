from graphviz import Digraph

dot = Digraph(comment='Use Case Diagram')

# Actor
dot.node('User', 'Vartotojas', shape='oval')

# Use cases
dot.node('UC1', 'Įkelti signalą', shape='ellipse')
dot.node('UC2', 'Apdoroti signalą', shape='ellipse')
dot.node('UC3', 'Aptikti priepuolį', shape='ellipse')
dot.node('UC4', 'Peržiūrėti rezultatus', shape='ellipse')
dot.node('UC5', 'Išsaugoti ataskaitą', shape='ellipse')

# Relationships
dot.edge('User', 'UC1')
dot.edge('User', 'UC4')
dot.edge('UC1', 'UC2')
dot.edge('UC2', 'UC3')
dot.edge('UC3', 'UC5')

dot.render('use_case_diagram', format='png')

