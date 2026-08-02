import streamlit as st
import google.generativeai as genai

# ページ基本設定
st.set_page_config(page_title="Gemini 画像生成アプリ", page_icon="🦖", layout="centered")

st.title("🦖 怪獣ジェネレーター (Streamlit + Gemini)")

# 1. 画像データを保存する箱（session_state）を用意する
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None

# 2. ボタンを押した時だけAPIを呼び出して画像を保存する
if st.button("怪獣を生成する", type="primary"):
    with st.spinner("怪獣を生成中..."):
        # Gemini APIの呼び出し
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("かっこいいファイアドラゴン怪獣のプロンプト...")
        
        # 生成された画像データを session_state に保存！
        st.session_state.generated_image = response

# 3. 保存された画像があれば表示する（画面操作や再描画でも画像が消えない）
if st.session_state.generated_image is not None:
    st.image(st.session_state.generated_image, caption="生成された怪獣", width=400)
    st.success("怪獣の生成が完了しました！")
