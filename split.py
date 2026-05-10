import os
import string
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def batch_generate_wordclouds(input_folder, output_folder):
    # 1. 创建输出文件夹，如果不存在的话
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ 已创建输出文件夹: {output_folder}")

    # 2. 定义英文停用词 (基础版，防止生成 too, very, also 这种无意义词)
    # 实际项目中建议加载 nltk.corpus.stopwords
    base_stopwords = set([
        'the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'for', 'it', 'as', 'was', 'with', 'on', 'by', 'at', 'an',
        'be', 'this', 'which', 'from', 'or', 'are', 'but', 'not', 'have', 'has', 'had', 'been', 'were', 'will', 'would',
        'could', 'should', 'can', 'may', 'might', 'must', 'shall', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'any', 'both', 'either', 'neither',
        'another', 'however', 'therefore', 'thus', 'hence', 'moreover', 'furthermore', 'additionally', 'consequently',
        'accordingly', 'meanwhile', 'otherwise', 'instead', 'nevertheless', 'nonetheless', 'regardless', 'despite',
        'although', 'though', 'while', 'whereas', 'if', 'unless', 'until', 'whether', 'because', 'since', 'even',
        'still', 'yet', 'already', 'almost', 'nearly', 'quite', 'rather', 'fairly', 'pretty', 'enough', 'what',
        'whatever', 'whichever', 'who', 'whom', 'whose', 'whoever', 'whomever', 'i', 'me', 'my', 'myself', 'we', 'our',
        'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'am', 'being',
        'having', 'doing', 'does', 'did', 'done', 'i\'m', 'you\'re', 'he\'s', 'she\'s', 'it\'s', 'we\'re', 'they\'re',
        'i\'ve', 'you\'ve', 'we\'ve', 'they\'ve', 'i\'ll', 'you\'ll', 'he\'ll', 'she\'ll', 'we\'ll', 'they\'ll', 'i\'d',
        'you\'d', 'he\'d', 'she\'d', 'we\'d', 'they\'d', 'let\'s', 'don\'t', 'doesn\'t', 'didn\'t', 'won\'t',
        'wouldn\'t', 'shan\'t', 'shouldn\'t', 'can\'t', 'couldn\'t', 'mustn\'t', 'isn\'t', 'aren\'t', 'wasn\'t',
        'weren\'t', 'haven\'t', 'hasn\'t', 'hadn\'t', 'needn\'t', 'daren\'t', 'mightn\'t', 'oughtn\'t', 'report',
        'annual', 'year', 'fiscal', 'ended', 'december', 'company', 'corporation', 'inc', 'limited', 'plc', 'group',
        'business', 'financial', 'statement', 'note', 'page', 'section', 'part', 'item', 'table', 'figure', 'see',
        'refer', 'include', 'including', 'based', 'related', 'pursuant', 'according', 'respectively', 'approximately',
        'million', 'billion', 'thousand', 'percent', 'rate', 'amount', 'total', 'net', 'gross', 'income', 'expense',
        'profit', 'loss', 'earnings', 'revenue', 'sales', 'cost', 'margin', 'ratio', 'share', 'stock', 'equity',
        'asset', 'liability', 'debt', 'capital', 'cash', 'flow', 'fund', 'investment', 'portfolio', 'risk', 'return',
        'growth', 'performance', 'result', 'operation', 'activity', 'strategy', 'objective', 'goal', 'plan', 'program',
        'initiative', 'project', 'development', 'research', 'technology', 'product', 'service', 'solution', 'platform',
        'system', 'process', 'procedure', 'policy', 'practice', 'standard', 'regulation', 'compliance', 'governance',
        'management', 'leadership', 'team', 'employee', 'personnel', 'staff', 'customer', 'client', 'user', 'consumer',
        'market', 'industry', 'sector', 'segment', 'region', 'geography', 'country', 'state', 'city', 'location',
        'office', 'facility', 'site', 'branch', 'subsidiary', 'affiliate', 'partner', 'vendor', 'supplier',
        'distributor', 'agent', 'representative', 'consultant', 'advisor', 'auditor', 'lawyer', 'accountant', 'banker',
        'investor', 'shareholder', 'stakeholder', 'board', 'director', 'officer', 'executive', 'chairman', 'ceo', 'cfo',
        'coo', 'president', 'vice', 'senior', 'junior', 'chief', 'head', 'manager', 'supervisor', 'coordinator',
        'specialist', 'analyst', 'engineer', 'developer', 'designer', 'architect', 'scientist', 'researcher',
        'professor', 'doctor', 'nurse', 'teacher', 'student', 'trainee', 'intern', 'apprentice', 'volunteer',
        'contractor', 'freelancer', 'expert', 'professional', 'practitioner', 'technician', 'operator', 'worker',
        'laborer', 'workforce', 'human', 'resource', 'talent', 'skill', 'competency', 'capability', 'capacity',
        'ability', 'potential', 'productivity', 'efficiency', 'effectiveness', 'quality', 'quantity', 'volume', 'value',
        'price', 'benefit', 'advantage', 'disadvantage', 'strength', 'weakness', 'opportunity', 'threat', 'challenge',
        'issue', 'problem', 'answer', 'response', 'reaction', 'feedback', 'comment', 'suggestion', 'recommendation',
        'proposal', 'idea', 'concept', 'theory', 'hypothesis', 'assumption', 'prediction', 'forecast', 'projection',
        'estimate', 'calculation', 'measurement', 'metric', 'indicator', 'benchmark', 'target', 'aim', 'purpose',
        'intention', 'motivation', 'incentive', 'reward', 'punishment', 'penalty', 'sanction', 'discipline', 'control',
        'monitoring', 'supervision', 'oversight', 'review', 'audit', 'inspection', 'examination', 'investigation',
        'analysis', 'evaluation', 'assessment', 'appraisal', 'judgment', 'decision', 'choice', 'selection', 'option',
        'alternative', 'possibility', 'probability', 'likelihood', 'chance', 'uncertainty', 'volatility', 'variability',
        'fluctuation', 'change', 'shift', 'transition', 'transformation', 'evolution', 'progress', 'advancement',
        'improvement', 'enhancement', 'upgrade', 'update', 'revision', 'modification', 'adjustment', 'adaptation',
        'customization', 'personalization', 'localization', 'globalization', 'internationalization', 'standardization',
        'normalization', 'optimization', 'maximization', 'minimization', 'rationalization', 'centralization',
        'decentralization', 'integration', 'consolidation', 'diversification', 'specialization', 'differentiation',
        'segmentation', 'targeting', 'positioning', 'branding', 'marketing', 'advertising', 'promotion', 'distribution',
        'logistics', 'supply', 'chain', 'procurement', 'purchasing', 'sourcing', 'outsourcing', 'insourcing',
        'offshoring', 'nearshoring', 'reshoring', 'backshoring', 'onshoring', 'homeshoring', 'telecommuting', 'remote',
        'work', 'virtual', 'collaboration', 'cooperation', 'coordination', 'communication', 'interaction', 'engagement',
        'participation', 'involvement', 'contribution', 'commitment', 'dedication', 'loyalty', 'retention', 'turnover',
        'attrition', 'recruitment', 'hiring', 'onboarding', 'training', 'learning', 'education', 'coaching',
        'mentoring', 'counseling', 'guidance', 'support', 'assistance', 'help', 'service', 'care', 'welfare',
        'wellbeing', 'health', 'safety', 'security', 'protection', 'prevention', 'mitigation', 'reduction',
        'elimination', 'avoidance', 'handling', 'treatment', 'resolution', 'settlement', 'conciliation', 'mediation',
        'arbitration', 'litigation', 'prosecution', 'defense', 'appeal', 'reconsideration', 'reexamination',
        'reassessment', 'reevaluation', 'reappraisal', 'reanalysis', 'reinterpretation', 'reconstruction',
        'reorganization', 'restructuring', 'reengineering', 'redesign', 'refurbishment', 'renovation', 'restoration',
        'rehabilitation', 'recovery', 'recuperation', 'convalescence', 'healing', 'cure', 'remedy', 'therapy',
        'medication', 'drug', 'pharmaceutical', 'biotechnology', 'medical', 'device', 'equipment', 'instrument', 'tool',
        'apparatus', 'machine', 'mechanism', 'network', 'infrastructure', 'framework', 'architecture', 'structure',
        'organization', 'institution', 'establishment', 'enterprise', 'venture', 'undertaking', 'endeavor', 'effort',
        'attempt', 'try', 'trial', 'test', 'experiment', 'pilot', 'prototype', 'model', 'sample', 'specimen', 'example',
        'instance', 'case', 'situation', 'circumstance', 'condition', 'state', 'status', 'position', 'place', 'spot',
        'point', 'area', 'zone', 'district', 'neighborhood', 'community', 'locality', 'vicinity', 'surroundings',
        'environment', 'setting', 'context', 'background', 'history', 'past', 'present', 'future', 'time', 'period',
        'duration', 'interval', 'span', 'range', 'scope', 'extent', 'scale', 'size', 'dimension', 'magnitude', 'degree',
        'level', 'grade', 'rank', 'proportion', 'percentage', 'fraction', 'part', 'portion', 'division', 'unit',
        'component', 'element', 'factor', 'aspect', 'feature', 'characteristic', 'attribute', 'property', 'trait',
        'nature', 'essence', 'substance', 'material', 'matter', 'thing', 'object', 'article', 'goods', 'commodity',
        'merchandise', 'inventory', 'reserve', 'store', 'cache', 'hoard', 'collection', 'accumulation', 'aggregation',
        'compilation', 'assembly', 'gathering', 'meeting', 'conference', 'convention', 'symposium', 'seminar',
        'workshop', 'forum', 'panel', 'discussion', 'debate', 'dialogue', 'conversation', 'talk', 'speech',
        'presentation', 'lecture', 'address', 'remark', 'observation', 'memo', 'declaration', 'announcement',
        'proclamation', 'publication', 'release', 'issuance', 'circulation', 'dissemination', 'broadcast',
        'transmission', 'message', 'signal', 'information', 'data', 'fact', 'detail', 'particular', 'specific',
        'general', 'overall', 'aggregate', 'combined', 'joint', 'mutual', 'common', 'shared', 'collective', 'corporate',
        'organizational', 'institutional', 'systemic', 'structural', 'functional', 'operational', 'administrative',
        'managerial', 'executive', 'strategic', 'tactical', 'technical', 'technological', 'scientific', 'innovative',
        'creative', 'original', 'unique', 'distinctive', 'special', 'particular', 'individual', 'personal', 'private',
        'confidential', 'proprietary', 'exclusive', 'restricted', 'limited', 'controlled', 'regulated', 'monitored',
        'supervised', 'overseen', 'managed', 'administered', 'governed', 'directed', 'guided', 'led', 'organized',
        'arranged', 'planned', 'scheduled', 'timed', 'paced', 'phased', 'staged', 'stepped', 'gradual', 'progressive',
        'incremental', 'cumulative', 'compound', 'complex', 'complicated', 'intricate', 'elaborate', 'sophisticated',
        'advanced', 'mature', 'developed', 'evolved', 'refined', 'polished', 'perfected', 'balanced', 'stable',
        'steady', 'consistent', 'reliable', 'dependable', 'trustworthy', 'credible', 'reputable', 'respected',
        'esteemed', 'valued', 'appreciated', 'recognized', 'acknowledged', 'honored', 'awarded', 'praised', 'commended',
        'applauded', 'celebrated', 'acclaimed', 'renowned', 'famous', 'noted', 'notable', 'remarkable', 'outstanding',
        'exceptional', 'extraordinary', 'impressive', 'striking', 'stunning', 'amazing', 'astonishing', 'astounding',
        'surprising', 'unexpected', 'unforeseen', 'unanticipated', 'unpredicted', 'unprojected', 'unestimated',
        'uncalculated', 'unmeasured', 'unassessed', 'unevaluated', 'unappraised', 'unanalyzed', 'unexamined',
        'uninvestigated', 'unexplored', 'unresearched', 'unstudied', 'untested', 'untried', 'unproven', 'unverified',
        'unconfirmed', 'unvalidated', 'unsubstantiated', 'unsupported', 'unfounded', 'baseless', 'groundless',
        'unjustified', 'unwarranted', 'unreasonable', 'illogical', 'irrational', 'absurd', 'ridiculous', 'preposterous',
        'nonsensical', 'foolish', 'silly', 'stupid', 'dumb', 'ignorant', 'uninformed', 'unaware', 'unconscious',
        'unmindful', 'unthinking', 'unreflective', 'unconsidered', 'uncontemplated', 'unmeditated', 'unpondered',
        'unruminated', 'undigested', 'unassimilated', 'unabsorbed', 'unincorporated', 'unintegrated', 'uncombined',
        'unmixed', 'unblended', 'unmerged', 'unfused', 'ununited', 'unjoined', 'unconnected', 'unlinked', 'unrelated',
        'unassociated', 'unaffiliated', 'unaligned', 'unattached', 'unbound', 'unfastened', 'unsecured', 'unprotected',
        'undefended', 'unguarded', 'unwatched', 'unmonitored', 'unsupervised', 'uncontrolled', 'unregulated',
        'unrestricted', 'unlimited', 'unbounded', 'unconfined', 'unconstrained', 'uninhibited', 'unrestrained',
        'unbridled', 'unchecked', 'unopposed', 'unchallenged', 'uncontested', 'undisputed', 'unquestioned',
        'unquestionable', 'indisputable', 'irrefutable', 'incontrovertible', 'undeniable', 'unmistakable',
        'unambiguous', 'clear', 'obvious', 'evident', 'apparent', 'manifest', 'plain', 'patent', 'distinct', 'definite',
        'certain', 'sure', 'confident', 'assured', 'positive', 'absolute', 'complete', 'entire', 'whole', 'full',
        'comprehensive', 'exhaustive', 'thorough', 'detailed', 'meticulous', 'careful', 'cautious', 'prudent',
        'judicious', 'sensible', 'rational', 'coherent', 'sound', 'valid', 'legitimate', 'legal', 'lawful',
        'authorized', 'approved', 'sanctioned', 'endorsed', 'backed', 'sponsored', 'funded', 'financed', 'invested',
        'subsidized', 'granted', 'allocated', 'appropriated', 'designated', 'assigned', 'delegated', 'entrusted',
        'committed', 'dedicated', 'devoted', 'pledged', 'promised', 'guaranteed', 'warranted', 'insured', 'safeguarded',
        'defended', 'guarded', 'watched', 'monitored', 'supervised', 'overseen', 'managed', 'administered', 'governed',
        'directed', 'guided', 'led', 'controlled', 'regulated', 'restricted', 'limited', 'constrained', 'confined',
        'bounded'
    ])

    # 获取所有txt文件
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    print(f"📂 发现 {len(txt_files)} 个文件，准备开始批量生成...")

    for i, filename in enumerate(txt_files):
        file_path = os.path.join(input_folder, filename)
        print(f"[{i + 1}/{len(txt_files)}] 正在处理: {filename}...")

        try:
            # 读取内容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 英文预处理
            content = content.lower()  # 转小写
            # 去除标点符号
            content = content.translate(str.maketrans('', '', string.punctuation))
            # 分词
            words = content.split()

            # 过滤停用词和短词
            filtered_words = [w for w in words if w not in base_stopwords and len(w) > 2]

            if not filtered_words:
                print(f"  ⚠️ 警告: {filename} 过滤后没有有效词汇，跳过。")
                continue

            # 统计词频 (用于生成词云)
            word_counts = Counter(filtered_words)

            # 生成词云
            wc = WordCloud(
                background_color='white',
                width=1200,
                height=800,
                max_words=100,  # 每个文件只显示前100个高频词，保持整洁
                random_state=42,  # 固定随机种子，让颜色布局每次一样
                colormap='viridis'  # 颜色主题
            )

            wc.generate_from_frequencies(word_counts)

            # 保存图片
            # 将文件名后缀改为 .png
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_folder, output_filename)

            wc.to_file(output_path)
            print(f"  ✅ 已保存: {output_filename}")

        except Exception as e:
            print(f"  ❌ 处理 {filename} 时出错: {e}")

    print("\n🎉 全部完成！请查看文件夹:", output_folder)


# --- 运行配置 ---
# 输入文件夹：你存放拆分后txt的文件夹
input_dir = 'pdf_sections'
# 输出文件夹：生成的图片将存放在这里
output_dir = 'wordcloud_output'

# 检查输入文件夹是否存在
if os.path.exists(input_dir):
    batch_generate_wordclouds(input_dir, output_dir)
else:
    print(f"❌ 错误：找不到输入文件夹 '{input_dir}'，请先运行上一步拆分代码。")