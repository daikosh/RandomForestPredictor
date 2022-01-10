import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# ページ設定
st.set_page_config(
    page_title="Random Forest Predictor ver.1.0",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlitのメニューやフッターを隠す
hide_streamlit_style = """
    <style>
        MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# メインクラス
class RandomForestPredictor(object):
    def __init__(self):
        pass

    def draw_heatmap(self, df, annot):
        fig = plt.figure()
        sns.heatmap(df, vmin=-1, vmax=1, center=0, cmap='bwr', annot=annot)
        st.pyplot(fig)

    def draw_bar(self, df):
        x = df.columns
        y = df.iloc[0]
        fig = plt.figure()
        ax = sns.barplot(x, y)
        ax.set(ylim=(0, 1))
        st.pyplot(fig)

    def open(self):
        """ページの表示"""

        # タイトルの表示
        st.title("Random Forest Predictor  ver.1.0")
        st.write("ランダムフォレスト (回帰) を手軽に体験できるWebアプリケーションです。([参考URL](https://qiita.com/Kenta_Hoji/items/4036c248165b663daff4))")
        st.write("①データの可視化、②ハイパーパラメータの設定、③学習、④精度評価ができます。")
        st.write("左のサイドバーからcsvファイルをアップロードしてご使用ください。")
        st.write("※ 量的データにのみ対応しております。例) [Red Wine Quality (Kaggle)](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)")
        
        st.sidebar.header("0. データの読み込み")
        # アップロードウィジェットをサイドバーに表示
        uploaded_file = st.sidebar.file_uploader("Upload a csv file / csvファイルをアップロード")

        # ファイルがアップロードされている場合
        if uploaded_file is not None:
            # アップロードされたファイルのインポート
            df_uploaded = pd.read_csv(uploaded_file)

            # データの表示
            st.subheader("1. データの確認・可視化")
            with st.expander("Uploaded Data"):
                st.dataframe(df_uploaded)
                st.markdown(df_uploaded.shape)

            with st.expander("基本統計量"):
                st.dataframe(df_uploaded.describe())

            with st.expander("欠損値"):
                st.dataframe(df_uploaded.isnull().sum())

            with st.expander("相関係数"):
                st.dataframe(df_uploaded.corr())
                st.subheader("ヒートマップ")
                heatmap_annot = st.checkbox('ヒートマップに数値を表示')
                self.draw_heatmap(df_uploaded.corr(), heatmap_annot)

            with st.expander("目的変数の選択", expanded=True):
                target = st.selectbox(
                    '目的変数を選択してください',
                    df_uploaded.columns,
                )

            st.subheader("2. ハイパーパラメータの設定")
            # 説明変数と目的変数の設定
            x = df_uploaded.drop(target,axis=1)
            y = df_uploaded[target]

            with st.expander("検証用データの割合"):
                # 検証用データの割合を設定
                TEST_SIZE  = st.slider('検証用データの割合を設定', 0.1, 0.5, 0.3)

            # 学習データと評価データを作成
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=TEST_SIZE,
                random_state=42,
            )

            with st.expander("ランダムフォレストのハイパーパラメータの設定"):
                n_estimators = st.slider("n_estimators", 1, 1000, 100)
                max_depth = st.slider("max_depth", 1, 100, 10)
                criterion = st.radio("criterion", ["mse", "mae", "poisson"])
                bootstrap = st.radio("bootstrap", ["True", "False"])

            _, center, _ = st.columns(3)
            if center.button('学習開始'):
                # モデルの学習
                rf_reg = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    criterion=criterion,
                    bootstrap=bootstrap,
                )
                rf_reg.fit(x_train, y_train)

                # 予測
                y_pred = rf_reg.predict(x_test)

                # 精度評価
                st.subheader("3. 精度評価")
                df_scores = pd.DataFrame({
                    "R2": r2_score(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "MSE": mean_squared_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))},
                    index=["scores"],
                )

                st.dataframe(df_scores)
                self.draw_bar(df_scores)

        # ファイルがアップロードされていない場合
        else:
            # 警告の表示
            st.warning("Please upload a csv file.")

        # フッターの表示
        st.write("Copyright © daikosh 2022. All Rights Reserved.")
        st.sidebar.write("Copyright © daikosh 2022. All Rights Reserved.")


if __name__ == "__main__":
    # インスタンスの生成
    predictor = RandomForestPredictor()
    predictor.open()
