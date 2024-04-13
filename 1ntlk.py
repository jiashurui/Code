from nltk.book import *

text1.concordance("former")
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
fdist1 = FreqDist(text1);fdist1.plot(50, cumulative=True)