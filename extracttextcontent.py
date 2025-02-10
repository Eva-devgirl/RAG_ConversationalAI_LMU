from bs4 import BeautifulSoup

import os

input_path = './rawdatahtml/'
allhtmls = os.listdir(input_path)

output_path = './datei.html/'

for htmldata in allhtmls:
	print("Processing : ",htmldata)
	with open(input_path+htmldata, 'r', encoding='utf-8') as file:
		html_content = file.read()
	soup = BeautifulSoup(html_content, 'html.parser')
	text_content = soup.get_text()
	with open(output_path+htmldata[:-5], 'a', encoding='utf-8') as result:
		result.write(text_content)
