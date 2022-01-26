file1 = open("./test/GCDC2_test.csv", "a", encoding="utf-8")
 
file2 = open("./test/Enron_test.csv", "r", encoding="utf-8")
 
 
for line in file2:
 
  file1.write(line)
 
 
file1.close()
 
file2.close()