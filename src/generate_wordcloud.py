#%%
from pathlib import Path
this_file = Path(__file__).resolve()
this_directory = this_file.parent
project_directory = this_directory.parent
input_directory = project_directory/"texts"
output_directory = project_directory/"wordclouds"
#%%
with open(this_directory/"stop_words.txt") as f:
    stopwords = [i.strip() for i in f.readlines()]
stopwords[-5:]
#%%
# https://zhuanlan.zhihu.com/p/353795160
import tqdm, time
import jieba, wordcloud
skip_exists = True
bar = tqdm.tqdm(input_directory.glob("**/*.txt"), colour="yellow", leave=True)
for txt_file in bar:
    # 进度条
    bar.set_description_str(f"正在分析文件{txt_file.name}")
    # 目标路径设置
    path = txt_file.parent
    relative = path.relative_to(input_directory)
    new_path = output_directory/relative
    new_path.mkdir(exist_ok=True, parents=True)
    new_file = new_path/f"{txt_file.stem}.png"
    if new_file.exists() and skip_exists:
        continue
    # 读取文件内容
    with open(txt_file) as f:
        s = f.read()
    # 分词
    ls = jieba.lcut(s) # 生成分词列表
    text = ' '.join(ls) # 连接成字符串
    # 词云
    stopwords_new = ["的","是","了", 
                 "就是", "这个", "然后",
                 "说", "想", "做",
                 "希望","是因为", "东西", 
                 "采访者1", "采访者2", "受访者", "采访者"]+stopwords
    stopwords_new = list(set(stopwords_new))
    wc = wordcloud.WordCloud(font_path="msyh.ttc",
                         width = 1000,
                         height = 700,
                         background_color='white',
                         max_words=100,stopwords=stopwords_new)
    # msyh.ttc电脑本地字体，写可以写成绝对路径
    wc.generate(text) # 加载词云文本
    wc.to_file(new_file) # 保存词云文件
    
    
# %%
