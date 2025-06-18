import streamlit as st
import pandas as pd
import io
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer
from zipfile import ZipFile

janome_tokenizer = Tokenizer()

# ✅ 僅保留語意上有價值的詞：名詞、動詞、形容詞
def tokenize_japanese(text):
    tokens = []
    for token in janome_tokenizer.tokenize(text):
        part = token.part_of_speech.split(',')[0]
        if part in ['名詞', '動詞', '形容詞']:
            tokens.append(token.surface)
    return tokens

def extract_date_from_filename(filename):
    match = re.match(r'^([0-9]{4}(Q[1-4])?)', filename)
    return match.group(1) if match else "未知"

def compute_tfidf(sentences):
    vectorizer = TfidfVectorizer(tokenizer=tokenize_japanese)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return vectorizer, tfidf_matrix

def extract_keywords(tfidf_matrix, vectorizer, top_n=5):
    feature_names = vectorizer.get_feature_names_out()
    keywords = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[::-1]
        selected = [feature_names[idx] for idx in top_indices if row[idx] > 0][:top_n]
        keywords.append("、".join(selected) if selected else "（無）")
    return keywords

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

# Streamlit app
st.title("📊 TF-IDF 關鍵詞分析工具（排除語助詞與標點）")
uploaded_files = st.file_uploader("請上傳 Excel 檔案（可多選）", type=["xlsx"], accept_multiple_files=True)

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
                    st.warning(f"{filename} 無有效語句，已略過")
                    continue

                vectorizer, tfidf_matrix = compute_tfidf(sentences)
                keywords = extract_keywords(tfidf_matrix, vectorizer)

                df["關鍵詞(TFIDF)"] = keywords
                df.insert(0, "資料日期", date_tag)

                buffer = io.BytesIO()
                df.to_excel(buffer, index=False)
                output_files.append((f"{filename[:-5]}_tfidf.xlsx", buffer))

                df["_來源檔名"] = filename
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                all_sentences.extend(sentences)

            if not all_sentences:
                st.error("無有效語句可分析")
                st.stop()

            vectorizer, tfidf_matrix = compute_tfidf(all_sentences)
            keywords = extract_keywords(tfidf_matrix, vectorizer)
            combined_df["關鍵詞(TFIDF)"] = keywords
            tfidf_summary_df = build_tfidf_summary(tfidf_matrix, vectorizer)

            try:
                sorted_dates = sorted([int(d[:4]) for d in all_dates if d[:4].isdigit()])
                min_date, max_date = str(sorted_dates[0]), str(sorted_dates[-1])
            except:
                sorted_dates = sorted(all_dates)
                min_date, max_date = sorted_dates[0], sorted_dates[-1]

            merged_filename = f"tfidf_總表合併_{min_date}-{max_date}.xlsx"
            merged_buffer = io.BytesIO()
            with pd.ExcelWriter(merged_buffer, engine="openpyxl") as writer:
                combined_df.to_excel(writer, sheet_name="合併句子分析", index=False)
                tfidf_summary_df.to_excel(writer, sheet_name="TFIDF關鍵字總表", index=False)
            merged_buffer.seek(0)
            output_files.append((merged_filename, merged_buffer))

            zip_buffer = io.BytesIO()
            with ZipFile(zip_buffer, "w") as zipf:
                for fname, buffer in output_files:
                    buffer.seek(0)
                    zipf.writestr(fname, buffer.read())
            zip_buffer.seek(0)

        st.success("分析完成 ✅")
        st.download_button("📥 下載分析結果（ZIP 壓縮包）", zip_buffer, file_name="tfidf_outputs.zip")
