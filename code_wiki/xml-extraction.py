import re

dataset = "enwiki-latest-abstract.xml"

with open(dataset) as infile:
	count = 1	
	for line in infile:
		sent_list = re.findall(r'<abstract>([^|].*)</abstract>',str(line), re.S)
		if(len(sent_list)!=0):
			count += len(sent_list)
			for i, sent in enumerate(sent_list):
				sent = re.sub("[^a-zA-Z ]", "", sent)
				sent = re.sub("[ ]+", " ", sent)
				if len(str(sent)) > 100:
					open("wiki-sentences.txt","a").write(str(sent)+"\n\n")
