import pdfplumber
import re
import os


def split_pdf_to_txt(pdf_path, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取PDF
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    # 按章节分割（年报常见章节标题）
    sections = re.split(r'(ITEM [0-9]+\.|PART [A-Z]+|Item [0-9]+\.|PART [IV]+)', full_text)

    # 组合标题和内容
    current_title = "introduction"
    section_count = 0

    for i, part in enumerate(sections):
        part = part.strip()
        if not part:
            continue

        # 判断是否是标题
        if re.match(r'^(ITEM|PART|Item|Part)\s', part):
            current_title = part
            section_count += 1
        else:
            # 写入txt文件
            if part and len(part) > 100:  # 只保存内容较长的段落
                filename = f"{section_count:02d}_{current_title.replace(' ', '_').replace('.', '')}.txt"
                filepath = os.path.join(output_folder, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(part)
                print(f"保存: {filename}")


# 使用示例
split_pdf_to_txt("Annual Report 2025.pdf", "pdf_sections")