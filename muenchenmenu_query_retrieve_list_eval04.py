import pandas as pd
import numpy as np
import json

def ndcg(weightsideals, results, limitpos):
	data = {}
	for nodeid in results.keys():
		data[nodeid] = [weightsideals[nodeid][0], weightsideals[nodeid][1], results[nodeid]]
#	print(data)
	table = pd.DataFrame.from_dict(data, orient="index", columns=['weight', 'ideal', 'result'])
	print(table)
	table["dcg_i"]=table["weight"]/np.log2(table["ideal"]+1)
	table.loc[table["result"] > limitpos, "weight"] = 0 # exclude results of lower than limit position (i.e. higher position number)
	table["dcg_r"]=table["weight"]/np.log2(table["result"]+1)
	print(table)
	print(table["dcg_r"].sum())
	print(table["dcg_i"].sum())
	return table["dcg_r"].sum()/table["dcg_i"].sum()


import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-proj-DwikG6Qsk0rlnchK8czNT3BlbkFJoAgxLhHtFmyLAvg8j2nJ"
openai.api_key = os.environ["OPENAI_API_KEY"]

rawdatadir = "./rawdata"
indexpath = "./muenchenmenuindex_simple_filenames"
questions_weightsideal_path = "./muenchenmenu_questions_weights_null.json"

print("Loading questions and nodes")
with open(questions_weightsideal_path, "r") as f:
	questions_weightsideal = json.load(f)
queries=questions_weightsideal.keys()
print(queries)

questions_weightsideal["Which restaurants are family friendly?"]["7927ea4a-d866-427d-bae4-ea2b9f561141"] = [1,1]
questions_weightsideal["Which restaurants are family friendly?"]["a51b226d-7249-41da-9e3a-f6638d5f00db"] = [0.9,2]
questions_weightsideal["Which restaurants are family friendly?"]["32b047e4-c9bf-4a95-b941-2b3e4b0c84cf"] = [0.75,3]
questions_weightsideal["Which restaurants are family friendly?"]["fc838511-fc40-4785-aae5-3d5b3128ee02"] = [0.6,4]
questions_weightsideal["Which restaurants are family friendly?"]["68271238-6eb3-43a2-82d0-a481dc84e9a6"] = [0.6,5]
questions_weightsideal["Which restaurants are family friendly?"]["96a0ae03-d8a4-4d18-b368-0e432224c765"] = [0.5,6]
questions_weightsideal["Which restaurants are family friendly?"]["2dbb4f17-b42b-407e-b5b3-30ee0f620c31"] = [0.4,7]
questions_weightsideal["Which restaurants are family friendly?"]["83b08db7-4a55-4ba1-8ad7-a9be7e4ef916"] = [0.35,8]
questions_weightsideal["Which restaurants are family friendly?"]["20ed750f-8df2-4870-a488-02467a6fb79e"] = [0.3,9]

questions_weightsideal["Welche Lokale sind Familienfreundlich?"]["7927ea4a-d866-427d-bae4-ea2b9f561141"] = [1,1]
questions_weightsideal["Welche Lokale sind Familienfreundlich?"]["a51b226d-7249-41da-9e3a-f6638d5f00db"] = [0.9,2]
questions_weightsideal["Welche Lokale sind Familienfreundlich?"]["32b047e4-c9bf-4a95-b941-2b3e4b0c84cf"] = [0.75,3]
questions_weightsideal["Welche Lokale sind Familienfreundlich?"]["fc838511-fc40-4785-aae5-3d5b3128ee02"] = [0.6,4]
questions_weightsideal["Welche Lokale sind Familienfreundlich?"]["68271238-6eb3-43a2-82d0-a481dc84e9a6"] = [0.6,5]
questions_weightsideal["Welche Lokale sind Familienfreundlich?"]["96a0ae03-d8a4-4d18-b368-0e432224c765"] = [0.5,6]
questions_weightsideal["Welche Lokale sind Familienfreundlich?"]["2dbb4f17-b42b-407e-b5b3-30ee0f620c31"] = [0.4,7]
questions_weightsideal["Welche Lokale sind Familienfreundlich?"]["83b08db7-4a55-4ba1-8ad7-a9be7e4ef916"] = [0.35,8]
questions_weightsideal["Welche Lokale sind Familienfreundlich?"]["20ed750f-8df2-4870-a488-02467a6fb79e"] = [0.3,9]


questions_weightsideal["Welches ist das beste Indisch Restaurant?"]["6663e97f-3a56-4b98-a259-45a9e21a9c68"] = [1,1]
questions_weightsideal["Welches ist das beste Indisch Restaurant?"]["1509ba85-a3a2-46b9-989e-74df9d0ca45b"] = [0.8,2]
questions_weightsideal["Welches ist das beste Indisch Restaurant?"]["91723088-19f6-445d-866f-ee1c688232ba"] = [0.7,3]

questions_weightsideal["Wo kocht man nur mit Bio Produkten"]["37f86dd0-1585-48ac-bd82-a8a4118d591d"] = [1,1]
questions_weightsideal["Wo kocht man nur mit Bio Produkten"]["960ea2b4-62d1-4839-81e8-7660f4d4ee7a"] = [1,1]
questions_weightsideal["Wo kocht man nur mit Bio Produkten"]["1509ba85-a3a2-46b9-989e-74df9d0ca45b"] = [0.9,2]
questions_weightsideal["Wo kocht man nur mit Bio Produkten"]["83d01ccd-a12f-4d30-9238-a4d706b68e9b"] = [0.8,3]
questions_weightsideal["Wo kocht man nur mit Bio Produkten"]["e521b1f3-d808-4288-b2d1-6707acdb712a"] = [0.8,4]
questions_weightsideal["Wo kocht man nur mit Bio Produkten"]["6663e97f-3a56-4b98-a259-45a9e21a9c68"] = [0.7,5]

questions_weightsideal["In welchem Lokal kann man Firmen Events feiern?"]["96a0ae03-d8a4-4d18-b368-0e432224c765"] = [1,1]
questions_weightsideal["In welchem Lokal kann man Firmen Events feiern?"]["706965b7-ba67-488d-8f70-32720c3783e0"] = [0.8,2]
questions_weightsideal["In welchem Lokal kann man Firmen Events feiern?"]["706965b7-ba67-488d-8f70-32720c3783e0"] = [0.8,3]
questions_weightsideal["In welchem Lokal kann man Firmen Events feiern?"]["4a66f164-3127-4850-af78-5a5f4a767d63"] = [0.7,4]


questions_weightsideal["Wo isst man die beste Pasta?"]["8dd2b581-2bb7-4344-b086-a141758d2e22"] = [1,1]
questions_weightsideal["Wo isst man die beste Pasta?"]["5acf3c66-3f77-47d2-b3af-7cfd4bfabdb1"] = [0.9,2]
questions_weightsideal["Wo isst man die beste Pasta?"]["96a0ae03-d8a4-4d18-b368-0e432224c765"] = [0.8,3]
questions_weightsideal["Wo isst man die beste Pasta?"]["62234178-96c2-475f-b31c-92aa17357d50"] = [0.6,4]
questions_weightsideal["Wo isst man die beste Pasta?"]["d2ebf0ca-8592-454c-9e56-cc647cf5ad2d"] = [0.5,5]
questions_weightsideal["Wo isst man die beste Pasta?"]["610e3d60-dc59-45b4-86a8-60e674df7ae7"] = [0.35,6]
questions_weightsideal["Wo isst man die beste Pasta?"]["b3482d54-51dd-43fe-adcd-ae2f51f3bac8"] = [0.25,7]
questions_weightsideal["Wo isst man die beste Pasta?"]["56828ea5-b570-4064-bb74-324641279147"] = [0.15,8]

questions_weightsideal["Welches ist das beste Restaurant in Schwabing?"]["ad773053-fd05-42a7-b744-36ebe6d29135"] = [1,1]
questions_weightsideal["Welches ist das beste Restaurant in Schwabing?"]["8dd2b581-2bb7-4344-b086-a141758d2e22"] = [0.8,2]
questions_weightsideal["Welches ist das beste Restaurant in Schwabing?"]["4e93882c-d068-42de-8436-6e4de75bc825"] = [0.75,3]
questions_weightsideal["Welches ist das beste Restaurant in Schwabing?"]["1a4d98c9-0733-42cd-bc87-097d2c2fb5e5"] = [0.7,4]
questions_weightsideal["Welches ist das beste Restaurant in Schwabing?"]["348f6e9c-94a8-4be4-a824-e09140a59201"] = [0.65,5]
questions_weightsideal["Welches ist das beste Restaurant in Schwabing?"]["23feb893-d8b4-4d0d-931b-bf413435416a"] = [0.6,6]
questions_weightsideal["Welches ist das beste Restaurant in Schwabing?"]["2f94b0b8-1277-43e9-9071-9084baa6afb2"] = [0.6,7]









from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core import StorageContext, load_index_from_storage, SimpleDirectoryReader
# rebuild storage context
print("Rebuilding context.")
storage_context = StorageContext.from_defaults(persist_dir=indexpath)
print("Loading index.")
# load index
index = load_index_from_storage(storage_context)
print("Creating query engine.")
query_engine = index.as_query_engine()
print("Creating Retriever.")
retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=300)
print("Node IDs")
print(index.vector_store._data.embedding_dict.keys())
allnodes=index.vector_store._data.embedding_dict.keys()
print(len(allnodes))
print("Start querying.")
ndcg_results={}
response_results={}
for query in queries:
	print(query)
		
	nodelist=[]
	
	nodes = retriever.retrieve(query)
	for node in nodes:
		nodelist.append(node.node_id)
	nodedict = {}
	nodepos = 1
	for nodeid in nodelist:
		nodedict[nodeid]=nodepos
		nodepos=nodepos+1

	for i in range(0,6):
		print(nodelist[i])
	result = ndcg(questions_weightsideal[query], nodedict, 10)
	print("Result NDCG : ", result)
	response = query_engine.query(query)
	print(response)
	ndcg_results[query]=result
	response_results[query]=str(response)
	print("===========================")
with open("./eval04_ndcg_results.json", "w") as f:
	json.dump(ndcg_results, f)

with open("./eval04_response_results.json", "w") as f:
	json.dump(response_results, f)

reference_answers = {"Which restaurants are family friendly?": "Dies sind einige der besten Restaurants in München für Familien mit Kindern:Naxos TavernaTrattoria Da FaustoRestaurant Pils Corner", "Welche Lokale sind Familienfreundlich?": "Dies sind einige der besten Restaurants in München für Familien mit Kindern:Naxos TavernaTrattoria Da FaustoRestaurant Pils Corner", "Welches ist das beste Indisch Restaurant?": "Indian Mango,Das Restaurant bietet ein reichhaltiges Angebot an authentischen nord- und südindischen Gerichten, stets frisch zubereitet. Liebhaber indischer Küche finden neben den konventionellen Speisen auch eine gute Auswahl veganer Speisen. Die wechselnden Tagesspezialitäten halten weitere Überraschungen bereit.","Wo kocht man nur mit Bio Produkten": "Willkommen im La Fayette, einem erstklassigen Restaurant, das mit mediterraner Köstlichkeit und einem unvergleichlichen, designorientierten Ambiente Ihre Sinne verzaubert. Hier verschmelzen feinste Zutaten zu Kunstwerken auf dem Teller, während das elegante Interieur Ihre Erwartungen an exklusive Gastronomie übertrifft." ,  "In welchem Lokal kann man Firmen Events feiern?": "Sie können unsere Veranstaltung ‚Dinner in the Dark‘ auch für Ihre Firmen-, Weihnachts-, oder Familienfeier buchen und Ihren Freunden und Kollegen einen Abend bescheren, an den sie noch lange zurückdenken werden.  Tauchen Sie gemeinsam ein in eine völlig neue Sinneswahrnehmung und genießen Sie unser vorzüglich komponiertes 4-Gang-Menü.", "Wo isst man die beste Pasta?": "Um ehrlich zu sein, sind wir ja immer noch ein bisschen skeptisch in Sachen Schwabinger Tor. Kulinarisch ballert das Bauprojekt aber auf jeden Fall ordentlich raus. Eine ganz besondere Entdeckung: Die Marta Trattoria, die uns nicht nur mit der Einrichtung aus den Latschen haut, sondern auch sofort mit Dolce-Vita-Gefühl versorgt. Das geht ganz wunderbar mit neapolitanischer Pizza, die zum Beispiel mit Salsiccia und Steinpilzen belegt ist. Oder man probiert es mit Vitello Tonnato, Antipasti und hausgemachter Pasta. Dazu einen Vino oder Aperitivo und ihr könnt euch eure Urlaubstage sparen.", "Welches ist das beste Restaurant in Schwabing?" : "Das Ali Bey ist genau der richtige Laden für alle, die keinen Bock auf das Klischee vom Döner als türkisches Nationalgericht haben. In dem Laden in der Schraudolphstraße gibt es authentische Gerichte, die uns teilweise so fremd sind, dass wir uns am liebsten auf die Empfehlungen des Hauses verlassen würden. Toll finden wir die hauptsächlich fleischlosen Mezze, die ihr euch als Vorspeisen zusammenstellen könnt. Darunter zum Beispiel auch Şakşuka, die türkische Variante von Shakshuka, die ihr im Ali Bey als leckere Mischung aus Zucchini, Auberginen, Paprika und Spitzpaprika in fein abgeschmeckter Tomatensauce mit Knoblauch bekommt. Und dann gibt es natürlich noch jede Menge Hauptspeisen, von denen ihr euch einfach direkt im Laden mehr erzählen lassen solltet. Eines unserer Highlights: Künefe. Eine Nachspeise aus dünnen Teigfäden, Mozzarella, Pistazien und Zuckersirup!" }

#https://plainenglish.io/community/evaluating-nlp-models-a-comprehensive-guide-to-rouge-bleu-meteor-and-bertscore-metrics-d0f1b1

print("Importing response scorers.")

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge4', 'rougeL'], use_stemmer=True)

#from nltk.translate.bleu_score import sentence_bleu

from nltk.translate import meteor_score
from nltk import word_tokenize

import nltk
print("Downloading wordnet.")
nltk.download('wordnet', download_dir='./nltk_data')

print("Evaluating responses.")
eval_response_results = {}

from bleu_sentence_git import sentence_bleu # use program text copied from github because in nltk implementation, there is apparently an error

for question in reference_answers.keys():
	rouge_results = scorer.score(response_results[question], reference_answers[question])
	bleu_results = sentence_bleu([reference_answers[question].split()], response_results[question].split())
	tokenized_reference = word_tokenize(reference_answers[question])
	tokenized_candidate = word_tokenize(response_results[question])	
	meteor_results = meteor_score.meteor_score([tokenized_reference], tokenized_candidate)
	eval_response_results[question] = {"result_answer":response_results[question], "reference_answer":reference_answers[question], "ndcg":ndcg_results[question], "rouge":rouge_results, "meteor":meteor_results, "bleu":bleu_results}
	print(question, response_results[question], reference_answers[question], eval_response_results[question])

with open("./eval04_response_all_results.json", "w") as f:
	json.dump(eval_response_results, f)

print("----------------------------------")
print(eval_response_results)
