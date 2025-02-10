import json

questions = ("Which restaurants are family friendly?", "Welche Lokale sind Familienfreundlich?", "Welches ist das beste Indisch Restaurant?", "Wo kocht man nur mit Bio Produkten", "In welchem Lokal kann man Firmen Events feiern?", "Wo isst man die beste Pasta?", "Welches ist das beste Restaurant in Schwabing?")

with open('muenchenmenu_nodes.json', 'r') as f:
  data = json.load(f)

result={}
for question in questions:
	result[question]={}
	for item in data:
		print(item)
#	print(data[item]["text"],"\n")
		print(data[item]["metadata"])	
		print(data[item]["weightideal"])
		print(data[item]["count"])
		print("----------------------------------------------------")
		result[question][item]=data[item]["weightideal"]

with open("muenchenmenu_questions_weights.json", "w") as g:
	json.dump(result,g)


