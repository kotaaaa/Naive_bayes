import naive_bayes
import numpy

nb = naive_bayes.NaiveBayses('baseball0525.txt', 'soccer0525.txt','jpn')
nb.print_file1()
nb.print_file2()
print(nb.get_vocabulary())
