import naive_bayes
import numpy

nb = naive_bayes.NaiveBayses('baseball0525.txt','soccer0525.txt','jpn')
# nb.print_file1()
# nb.print_file2()
# print(nb.get_vocabulary())
# print(nb.split_words('国産ジェット旅客機「三菱リージョナルジェット（ＭＲＪ）」を開発する三菱航空機（愛知県豊山町）が欧州の空を泳いだ。'))
# print(nb.get_words_frequency_file1().shape)
# print(nb.get_words_frequency_file1())
print('Category is >>> ',nb.predict('大谷、８試合ぶり複数安打 ２安打２得点で勝利に貢献'))
print('Category is >>> ',nb.predict('サッカーＪ１ヴィッセル神戸にスペイン代表、イニエスタ選手の入団が決まり'))
