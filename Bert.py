import os
import re
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# --- 配置 ---
input_folder = 'important_part'
# 使用专门针对英文情感分析的 BERT 模型
# 这个模型在大量英文评论数据上训练过，效果很好
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

# --- 1. 加载预训练模型和分词器 ---
print(f"正在加载英文模型: {model_name} ... (首次运行需下载约400MB)")
try:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # 自动检测 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ 模型加载成功，运行在 {device} 上")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()


# --- 2. 定义英文分句函数 ---
def split_into_english_sentences(text):
    """
    按英文标点分割句子
    """
    # 使用正则表达式按 . ! ? 分割
    # 注意：英文中 . 也可能是缩写（如 U.S.A.），这里做简单处理，复杂情况建议用 nltk
    sentences = re.split(r'[.!?]+', text)
    # 去除首尾空格和空字符串
    return [s.strip() for s in sentences if s.strip()]


# --- 3. 定义情感分析函数 ---
def analyze_english_sentiment(text):
    """
    对英文句子进行情感分析
    该模型输出 1-5 星：1=很差(消极), 5=很好(积极)
    """
    # 截断过长的句子，防止报错
    if len(text) > 512:
        text = text[:512]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # 获取预测的类别概率
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # 获取预测的星级 (1到5)
        # argmax 返回的是 0-4 的索引，对应星级 1-5
        predicted_class_index = torch.argmax(probabilities, dim=-1).item()
        stars = predicted_class_index + 1
        confidence = probabilities[0, predicted_class_index].item()

    # 映射星级到情感标签
    # 5星 -> 积极
    # 4星 -> 积极
    # 3星 -> 中性
    # 2星 -> 消极
    # 1星 -> 消极
    if stars >= 4:
        sentiment = "积极"
    elif stars == 3:
        sentiment = "中性"
    else:
        sentiment = "消极"

    return sentiment, stars, confidence


# --- 4. 批量处理并生成报告 ---
def generate_english_reports(input_folder):
    if not os.path.exists(input_folder):
        print(f"❌ 错误：找不到文件夹 '{input_folder}'")
        return

    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    if not txt_files:
        print(f"❌ 错误：文件夹中未找到 .txt 文件")
        return

    print(f"📂 发现 {len(txt_files)} 个文件，开始英文情感分析...\n")

    for i, filename in enumerate(txt_files):
        file_path = os.path.join(input_folder, filename)
        print(f"--- 正在处理: {filename} ---")

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
            continue

        # 分割句子
        sentences = split_into_english_sentences(content)

        if not sentences:
            print("  ⚠️ 无有效句子，跳过。")
            continue

        print(f"  📝 提取到 {len(sentences)} 个句子，正在分析...")

        results = []
        for j, sentence in enumerate(sentences):
            sentiment, stars, confidence = analyze_english_sentiment(sentence)
            results.append({
                "sentence": sentence,
                "sentiment": sentiment,
                "stars": stars,
                "confidence": confidence
            })

            # 每20句打印一次进度，避免刷屏
            if (j + 1) % 20 == 0:
                print(f"    进度: {j + 1}/{len(sentences)}...")

        # --- 生成四维度报告 ---
        total = len(results)
        positive_count = sum(1 for r in results if r['sentiment'] == '积极')
        negative_count = sum(1 for r in results if r['sentiment'] == '消极')
        neutral_count = total - positive_count - negative_count

        # 计算平均置信度
        avg_conf = sum(r['confidence'] for r in results) / total

        # 计算平均星级 (可选，作为补充指标)
        avg_stars = sum(r['stars'] for r in results) / total

        print(f"\n{'=' * 60}")
        print(f"📊 英文情感分析报告: {filename}")
        print(f"{'=' * 60}")
        print(f"1. 句子总数:        {total}")
        print(f"2. 情感分布:        积极 {positive_count} | 中性 {neutral_count} | 消极 {negative_count}")
        print(f"3. 积极情绪占比:    {positive_count / total:.2%}")
        print(f"4. 平均置信度:      {avg_conf:.2%}")
        # 额外赠送：平均星级
        print(f"5. 平均星级(1-5):   {avg_stars:.2f} 星")
        print(f"{'=' * 60}\n")


# --- 运行 ---
generate_english_reports(input_folder)