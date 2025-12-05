# app.py
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()


# --------------------------------------------------
# LLM からの回答を取得する関数
#   - input_text: 入力テキスト
#   - expert_type: "A" or "B"（ラジオボタンの選択値）
# --------------------------------------------------
def get_expert_response(input_text: str, expert_type: str) -> str:
    # 専門家ごとのシステムメッセージ
    system_messages = {
        "A": (
            "あなたは一流のマラソンコーチです。"
            "市民ランナーから上級者まで、それぞれのレベルに合わせて、"
            "トレーニングメニュー、フォーム、ペース配分、レース戦略、"
            "ケガ予防、栄養・補給などについて、わかりやすく具体的にアドバイスしてください。"
            "専門用語は必要に応じて簡単に説明し、日本語で丁寧に回答してください。"
        ),
        "B": (
            "あなたは優秀な営業コンサルタントです。"
            "法人営業・個人営業問わず、営業戦略、商談の組み立て、ヒアリング、提案資料、"
            "クロージング、リレーション構築、KPI設計などに詳しい専門家として振る舞ってください。"
            "実務でそのまま使える具体例やトーク例も交え、日本語でわかりやすくアドバイスしてください。"
        ),
    }

    system_message = system_messages.get(expert_type, system_messages["A"])

    # LangChain で LLM を呼び出す
    # ※ OPENAI_API_KEY が環境変数か Streamlit の secrets に設定されている必要があります
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # お好みで別モデルに変更可
        temperature=0.7,
    )

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=input_text),
    ]

    response = llm.invoke(messages)  # .invoke が推奨
    return response.content


# --------------------------------------------------
# Streamlit アプリ本体
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="専門家に相談できるミニLLMアプリ",
        page_icon="💬",
        layout="centered",
    )

    st.title("💬 専門家に相談できるミニLLMアプリ")

    st.markdown(
        """
このアプリでは、テキストを入力して **LLM（大規模言語モデル）** に質問すると、  
選択した「専門家」の立場から回答してくれます。

### 🔍 できること
1. 相談したい専門家の種類をラジオボタンから選びます  
   - **A：マラソンコーチの専門家**  
   - **B：営業コンサルの専門家**
2. 下の入力フォームに質問や相談内容を入力します  
3. 「送信」ボタンを押すと、専門家としての回答が画面に表示されます  

※ このアプリは **Python 3.11 + Streamlit + LangChain + OpenAI API** で動作します。  
※ OpenAI の API キーの設定が別途必要です。
"""
    )

    # 1️⃣ 専門家の選択
    expert_option = st.radio(
        "相談したい専門家を選んでください：",
        ("A：マラソンコーチの専門家", "B：営業コンサルの専門家"),
    )
    expert_type = "A" if expert_option.startswith("A") else "B"

    # 2️⃣ 質問入力
    user_input = st.text_area(
        "相談内容を自由に入力してください：",
        height=150,
        placeholder=(
            "例）フルマラソンでサブ4を目指したいのですが、3ヶ月でどんな練習をすべきですか？\n"
            "例）新規開拓営業で、初回訪問のヒアリングで意識すべきポイントを教えてください。"
        ),
    )

    # 3️⃣ 送信ボタン
    if st.button("送信"):
        if not user_input.strip():
            st.warning("質問を入力してください。")
            return

        with st.spinner("専門家が回答を生成中です…"):
            try:
                answer = get_expert_response(user_input, expert_type)
            except Exception as e:
                st.error("LLM 呼び出し中にエラーが発生しました。API キーやネットワーク設定を確認してください。")
                st.exception(e)
                return

        st.subheader("🧑‍🏫 専門家からの回答")
        st.write(answer)


if __name__ == "__main__":
    main()