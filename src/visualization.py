from wordcloud import WordCloud
import matplotlib.pyplot as plt

def filter_words_by_tfidf(tfidf_dicts, threshold):
    """
    TF-IDFのしきい値を使って単語をフィルタリングする関数。
    
    Args:
        tfidf_dicts (list of dict): 各文書の単語ごとのTF-IDF値を含む辞書のリスト。
        threshold (float): フィルタリングするためのTF-IDFのしきい値。
    
    Returns:
        dict: フィルタリングされた単語とTF-IDF値の辞書。
    """
    filtered_words = {}
    for tfidf_dict in tfidf_dicts:
        for word, score in tfidf_dict.items():
            if score >= threshold:
                filtered_words[word] = score
    return filtered_words

def generate_wordcloud(words, font_path=None):
    """
    指定された単語の頻度に基づいてワードクラウドを生成し、表示します。

    Args:
        words (dict): 単語とその頻度を含む辞書。キーは単語、値は対応するTF-IDFスコアなどの頻度。
        font_path (str, optional): 使用するフォントのファイルパス。デフォルトはNone。

    Returns:
        None: この関数はプロットを表示するため、返り値はありません。
    """
    
    # ワードクラウドの生成
    wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate_from_frequencies(words)
    
    # プロット
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # 軸をオフにする
    plt.show()
