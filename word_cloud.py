from wordcloud import WordCloud
import jieba
from collections import Counter
# from imageio import imread
import matplotlib.pyplot as plt

"""获取文本内容"""
with open("济南的冬天.txt", "r", encoding="utf-8") as fp:
    content = fp.read()
words_temp = jieba.lcut(content)
words = []
"""读取停用词"""
with open("C:/停用词/哈工大停用词.txt", "r", encoding="utf-8") as fp:
    stopwords = [s.rstrip() for s in fp.readlines()]

"""去掉切分词语中的停用词"""
for w in words_temp:
    if w not in stopwords:
        words.append(w)

frequency = dict(Counter(words))  # 去停用词之后的词频统计结果

font = "C:/Fonts/AaMingYueJiuLinTian.ttf"
# mask_image = imread("20160303160528046.png")

wc = WordCloud(font_path=font,
               background_color="white")
               # mask=mask_image)

wc.fit_words(frequency)  # 基于前面的词频统计

plt.imshow(wc)
plt.axis("off")
plt.show()
# wc.to_file("C:/Users/lenovo/Desktop/pic/6.png")
