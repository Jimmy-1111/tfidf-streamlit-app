import streamlit as st
import pandas as pd
import io
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer
from zipfile import ZipFile
from datetime import datetime

# 初始化日文斷詞器
janome_tokenizer = Tokenizer()

# 擷取資料日期
def extract_date_from_filename(filename):
    match = re.match(r'^([0-9]{4}(Q[1-4])?)', filename)
    return match.group(1) if match else "未知"

# 日文斷詞
def tokenize_japanese(text):
    return [token.surface for token in janome_tokenizer.tokenize(text)]

# 執行 TF-IDF 分析
def compute_tfidf(sentences):
    vectorizer = TfidfVectorizer(tokenizer=tokenize_japanese)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return vectorizer, tfidf_matrix

# 抽出句子關鍵詞
def extract_keywords(tfidf_matrix, vectorizer, top_n=5):
    feature_names = vectorizer.get_feature_names_out()
    keywords = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[::-1]
        selected = [feature_names[idx] for idx in top_indices if row[idx] > 0][:top_n]
        keywords.append("、".join(selected) if selected else "（無）")
    return keywords

# 統計所有關鍵詞 TF-IDF 值
def build_tfidf_summary(tfidf_matrix, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    tfidf_array = tfidf_matrix.toarray()
    tf_sum = np.sum(tfidf_array, axis=0)
    tf_avg = np.mean(tfidf_array, axis=0)
    tf_count = np.count_nonzero(tfidf_array, axis=0)
    summary_df = pd.DataFrame({
        "詞彙（關鍵字）": feature_names,
        "TF-IDF（總和）": tf_sum,
        "TF-IDF（平均）": tf_avg,
        "出現次數": tf_count
    }).sort_values(by="TF-IDF（總和）", ascending=False)
    return summary_df

# Streamlit UI
st.title("📊 TF-IDF 關鍵詞分析工具")
uploaded_files = st.file_uploader("請上傳一個或多個 Excel 檔案（含語句內容欄位）", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    column_name = st.text_input("請輸入要分析的欄位名稱（預設為：語句內容）", value="語句內容")

    if st.button("開始分析"):
        with st.spinner("分析中..."):

            combined_df = pd.DataFrame()
            output_files = []
            all_dates = []
            all_sentences = []

            for file in uploaded_files:
                filename = file.name
                date_tag = extract_date_from_filename(filename)
                all_dates.append(date_tag)

                df = pd.read_excel(file)
                if column_name not in df.columns:
                    st.warning(f"{filename} 缺少欄位：{column_name}，已略過")
                    continue

                df = df.dropna(subset=[column_name])
                sentences = df[column_name].astype(str).tolist()
                if not sentences:
                    st.warning(f"{filename} 中無有效句子，已略過")
                    continue

                # 個別 TF-IDF
                vectorizer, tfidf_matrix = compute_tfidf(sentences)
                keywords = extract_keywords(tfidf_matrix, vectorizer)

                df["關鍵詞(TFIDF)"] = keywords
                df.insert(0, "資料日期", date_tag)

                # 儲存個別檔案
                output_buffer = io.BytesIO()
                df.to_excel(output_buffer, index=False)
                output_files.append((f"{filename[:-5]}_tfidf.xlsx", output_buffer))

                df["_來源檔名"] = filename
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                all_sentences.extend(sentences)

            # 合併總表 TF-IDF
            if not all_sentences:
                st.error("無有效句子可分析")
                st.stop()

            vectorizer, tfidf_matrix = compute_tfidf(all_sentences)
            keywords = extract_keywords(tfidf_matrix, vectorizer)
            combined_df["關鍵詞(TFIDF)"] = keywords

            tfidf_summary_df = build_tfidf_summary(tfidf_matrix, vectorizer)

            # 取得總表的日期區間
            try:
                sorted_dates = sorted([int(d[:4]) for d in all_dates if d[:4].isdigit()])
                min_date, max_date = str(sorted_dates[0]), str(sorted_dates[-1])
            except:
                sorted_dates = sorted(all_dates)
                min_date, max_date = sorted_dates[0], sorted_dates[-1]

            merged_filename = f"tfidf_總表合併_{min_date}-{max_date}.xlsx"
            combined_buffer = io.BytesIO()
            with pd.ExcelWriter(combined_buffer, engine='openpyxl') as writer:
                combined_df.to_excel(writer, sheet_name="合併句子分析", index=False)
                tfidf_summary_df.to_excel(writer, sheet_name="TFIDF關鍵字總表", index=False)
            combined_buffer.seek(0)
            output_files.append((merged_filename, combined_buffer))

            # 壓縮全部
            zip_buffer = io.BytesIO()
            with ZipFile(zip_buffer, "w") as zipf:
                for fname, buffer in output_files:
                    buffer.seek(0)
                    zipf.writestr(fname, buffer.read())
            zip_buffer.seek(0)

        st.success("分析與匯出完成！")
        st.download_button("📥 下載所有結果（ZIP 壓縮包）", zip_buffer, file_name="tfidf_outputs.zip")
