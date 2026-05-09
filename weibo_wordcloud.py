# -*- coding: utf-8 -*-
"""
目的：读取微博2025年报PDF，生成英文词云图
注意：请确保文件名是 'Annual Report 2025.pdf' 或者修改代码中的文件名
"""

import pdfplumber
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter


# 1. 读取PDF文件
def extract_pdf_text(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except FileNotFoundError:
        print(f"错误：找不到文件 {pdf_path}，请检查文件名是否正确。")
    return text


# 2. 数据清洗与词频统计
def clean_and_count_words(text):
    # 2.1 使用正则表达式提取单词 (只保留字母，长度大于2)
    # 这一步相当于英文的"分词"
    words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())

    # 2.2 定义"停用词" (这些词太常见了，对分析没帮助，直接扔掉)
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day',
        'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'own',
        'say', 'she', 'too', 'use', 'your', 'each', 'just', 'like', 'make', 'many', 'some', 'such', 'that', 'they',
        'this', 'what', 'when', 'where', 'which', 'while', 'with', 'would', 'about', 'after', 'being', 'could', 'first',
        'from', 'have', 'into', 'more', 'only', 'other', 'over', 'such', 'than', 'then', 'them', 'these', 'time',
        'very', 'will', 'also', 'back', 'even', 'find', 'know', 'look', 'long', 'must', 'part', 'take', 'them', 'well',
        'year', 'your', 'business', 'company', 'limited', 'according', 'including', 'following', 'respectively',
        'substantially', 'herein', 'thereof', 'whereof', 'pursuant', 'determine', 'relating', 'whether', 'subject',
        'thereon', 'report', 'annual', 'item', 'section', 'note', 'table', 'figure', 'weibo', 'we', 'our', 'us', 'they',
        'them', 'their', 'its', 'it', 'he', 'she', 'has', 'had', 'have', 'do', 'does', 'did', 'done', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only',
        'own', 'same', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
    }

    # 2.3 过滤掉停用词
    filtered_words = [word for word in words if word not in stop_words]

    # 2.4 统计词频
    word_freq = Counter(filtered_words)
    return word_freq


# 3. 生成词云图
def plot_wordcloud(word_freq):
    # 将词频字典转换为字符串，供词云库使用
    wordcloud_text = " ".join(word for word, freq in word_freq.items() for _ in range(freq))

    # 创建词云对象
    wc = WordCloud(
        background_color="white",  # 背景颜色
        width=1000,  # 图片宽度
        height=600,  # 图片高度
        max_words=150,  # 最多显示150个词
        colormap="tab20c",  # 颜色方案，适合商业图表
        random_state=42,  # 随机种子，保证每次运行形状一样
        font_path=None  # 使用默认字体 (如果是中文需指定字体路径)
    )

    # 生成词云
    wc.generate(wordcloud_text)

    # 绘图
    plt.figure(figsize=(15, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")  # 不显示坐标轴
    plt.title("Weibo 2025 Annual Report - Word Cloud", fontsize=20, pad=20)
    plt.tight_layout()
    plt.show()


# --- 主程序 ---
if __name__ == "__main__":
    # 步骤1：提取文本
    print("正在读取 PDF 文件...")
    pdf_text = extract_pdf_text("Annual Report 2025.pdf")

    if pdf_text:
        # 步骤2：清洗和统计
        print("正在清洗数据并统计词频...")
        word_frequency = clean_and_count_words(pdf_text)

        # 打印前10个高频词，让你看看结果
        print("\n--- 报告中的高频词 Top 10 ---")
        for word, freq in word_frequency.most_common(10):
            print(f"{word}: {freq}")

        # 步骤3：画图
        print("\n正在生成词云图...")
        plot_wordcloud(word_frequency)
    else:
        print("未提取到文本，请检查文件路径。")