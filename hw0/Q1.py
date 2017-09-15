import sys
#print(sys.argv)
word = open(sys.argv[1], "r+")
Q1_txt = open("Q1.txt", "w")
string = word.read()
word_list = string.split()
#print(word_list)
word_appearred_ever = []
word_output = []
idx = 0
for e in word_list:
	#print(e)
	if e not in word_appearred_ever:
		word_output.append((e, idx, word_list.count(e)))
		word_appearred_ever.append(e)
		idx += 1
#print(word_output)
for e in word_output:
	s = e[0] + " " + str(e[1]) + " " + str(e[2]) + "\n"
	Q1_txt.write(s)
word.close()
Q1_txt.close()