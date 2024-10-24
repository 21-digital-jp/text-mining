import re
from janome.tokenizer import Tokenizer
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text, noise_patterns=None):
    """
    指定されたノイズを除去する関数。

    Args:
        text (str): 対象のテキスト。
        noise_patterns (list): 除去するノイズのパターンのリスト。指定がない場合、ノイズ除去は行わない。

    Returns:
        str: ノイズが除去されたテキスト（ノイズパターンが指定されていない場合は元のテキスト）。
    """
    # ノイズパターンが指定されていない場合はそのまま返す
    if noise_patterns is None:
        return text

    # 各ノイズパターンを削除
    for pattern in noise_patterns:
        text = re.sub(pattern, ' ', text)

    return text.strip()  # 前後の空白を削除

def tokenize_text(text):
    """
    テキストをトークン化し、単語の原型と品詞を取得する関数。

    Args:
        text (str): 対象のテキスト。

    Returns:
        list: (単語の原型, 品詞) のタプルのリスト。
    """
    # Janomeのトークナイザーを初期化
    tokenizer = Tokenizer()
    
    # 単語に分割し、品詞と原型を取得
    result = []
    for token in tokenizer.tokenize(text):
        word = token.base_form  # 原型を取得
        part_of_speech = token.part_of_speech.split(',')[0]  # 品詞を取得（最初の部分）
        result.append((word, part_of_speech))
    
    return result

def normalize_text(word_pos_list, normalization_dict=None):
    """
    品詞付き単語リストの正規化を行う関数。
    
    Args:
        word_pos_list (list): 品詞付きの単語リスト（タプルのリスト）。
        normalization_dict (dict): 正規化用の辞書（キーが元の単語、値が統一する単語）。指定がない場合はスルー。
        
    Returns:
        list: 正規化された品詞付き単語リスト。
    """
    # 辞書が指定されていない場合はそのまま返す
    if normalization_dict is None:
        return word_pos_list

    normalized = []
    for word, pos in word_pos_list:
        # 辞書に存在する場合は統一、存在しない場合はそのまま
        normalized_word = normalization_dict.get(word, word)
        normalized.append((normalized_word, pos))  # 品詞はそのまま残す

    return normalized

def fetch_and_remove_stopwords(url, word_pos_list):
    """
    指定されたURLからストップワードを取得し、品詞付き単語リストから除去する関数。
    
    Args:
        url (str): ストップワードのリストが記載されているURL。
        word_pos_list (list): 品詞付きの単語リスト（タプルのリスト）。
        
    Returns:
        list: ストップワードが除去された品詞付き単語リスト。
    """
    # ストップワードを取得
    response = requests.get(url)
    stopwords = response.text.splitlines()
    
    # ストップワードを除去
    cleaned_word_pos_list = [(word, pos) for word, pos in word_pos_list if word not in stopwords]
    return cleaned_word_pos_list

def calculate_tfidf(word_pos_lists):
    """
    品詞付き単語リストのリストに基づいてTF-IDFを計算する関数。
    
    Args:
        word_pos_lists (list): 品詞付きの単語リストのリスト（タプルのリストのリスト）。
        
    Returns:
        dict: 単語をキー、TF-IDFを値とする辞書のリスト。
    """
    # 文書を生成するためのリスト
    docs = [' '.join(word for word, pos in word_pos_list if word.strip()) for word_pos_list in word_pos_lists]
    
    # TF-IDFベクトライザのインスタンスを作成
    vectorizer = TfidfVectorizer()
    
    # TF-IDFの計算
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    # 結果を辞書に格納
    feature_names = vectorizer.get_feature_names_out()
    
    # 各文書のTF-IDFスコアを辞書に変換
    tfidf_results = []
    for i in range(tfidf_matrix.shape[0]):
        tfidf_scores = tfidf_matrix.toarray()[i]
        tfidf_dict = {feature_names[j]: float(tfidf_scores[j]) for j in range(len(feature_names)) if tfidf_scores[j] > 0}
        tfidf_results.append(tfidf_dict)
    
    return tfidf_results

