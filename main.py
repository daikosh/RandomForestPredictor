import os
import datetime
import base64

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.model_selection import train_test_split

# 評価指標
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# パラメータの設定
TODAY = datetime.datetime.today().strftime("%Y%m%d_")


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

    def show_image(self, imgpath):
        """画像を表示"""
        if os.path.exists(imgpath):
            image = Image.open(imgpath)
            st.image(image, use_column_width="auto")

    def get_table_download_link(self, df):
        """ダウンロードリンクを取得"""
        nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        csv = df.to_csv(index=True)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}" download="{nowtime}.csv">Download</a>'
        return href

    def preprocess(self, df_uploaded):
        """前処理"""

        # 不要な特徴量の消去
        drop_features = [
            'Job Code', 'Project Name', 'Tag Number',\
            'Operation Frequency', 'HP', 'HP\nTotal',\
            'Small Nozzle Size', 'Remarks', 'h', 'h1',\
            'Propeller Diameter', 'Large Nozzle Size',\
            'Content Details', 'Content Kind',\
            'Floating \nRoof', 'Vendor', 'kW\nTotal',\
            'Viscosity', 'Operating Temperature', 'Tank ID',\
            'TEST DATA', 'Power Rating', 'kW', 'Tank Height'
            ]
        df = df_uploaded.drop(drop_features, axis=1)

        # カテゴリカル変数のダミー化 + NC_per_Qty列の作成
        fulldata = [df]
        for dataset in fulldata:
            dataset['Tank Type_CRT'] = dataset['Tank Type'].map(
                {'CRT': 1, 'FRT': 0, 'IFRT': 0, 'OTT': 0, 'DRT': 0, 'IFRT\n(DRT)': 0}
            )
            dataset['Tank Type_FRT'] = dataset['Tank Type'].map(
                    {'CRT': 0, 'FRT': 1, 'IFRT': 1, 'OTT': 1, 'DRT': 1, 'IFRT\n(DRT)': 1}
                )
            dataset['Content Type_1'] = dataset['Content Type'].map({1: 1, 2: 0, 3: 0})
            dataset['Content Type_2'] = dataset['Content Type'].map({1: 0, 2: 1, 3: 0})
            dataset['Purpose_1'] = dataset['Purpose'].map({1: 1, 2: 0, 3: 0})
            dataset['Purpose_2'] = dataset['Purpose'].map({1: 0, 2: 1, 3: 0})
            dataset['NC_per_Qty'] = dataset['Nominal Capacity']/dataset['Qty']

        # 不要な特徴量の消去
        df.drop(['Tank Type', 'Content Type', 'Purpose', 'Vendor Number', 'Nominal Capacity', 'Qty'], axis=1, inplace=True)

        return df

    def predict(self, df, modelpath):
        """予測モデルでy_predを予測する"""

        # 学習された予測モデルのインポート
        loaded_model = pickle.load(open(modelpath, 'rb'))

        # y_predを予測
        x = df
        y_pred = loaded_model.predict(x)

        return y_pred

    def create_df_results(self, df, y_pred):
        """df_resultsの作成"""

        # df_resultsの作成
        x = df
        df_results = pd.DataFrame(y_pred.tolist(), index=x.index, columns=['kW pred'])

        # y_predを平均値で区切って、Power Ratingに変換
        bins_ave = [-99, (2.2+3.7)/2, (3.7+5.5)/2, (5.5+7.5)/2, (7.5+11)/2, (11+15)/2,
                    (15+18)/2, (18+22)/2, (22+30)/2, (30+37)/2, (37+45)/2, (45+55)/2, 99]
        labels_unit_power = [2.2, 3.7, 5.5, 7.5, 11, 15, 18, 22, 30, 37, 45, 55]
        list_kw_to_power_rating = {2.2:1, 3.7:2, 5.5:3, 7.5:4, 11:5, 15:6, 18:7, 22:8, 30:9, 37:10, 45:11, 55:12}
        df_results['kW pred (round)'] = pd.cut(df_results['kW pred'], bins=bins_ave, labels=labels_unit_power).values
        df_results['Power Rating pred'] = df_results['kW pred (round)'].apply(list_kw_to_power_rating.__getitem__)

        return df_results

    def draw_pairplot(self, df):
        fig = plt.figure()
        sns.pairplot(df)
        st.pyplot(fig)

    def draw_heatmap(self, df, annot):
        fig = plt.figure()
        sns.heatmap(df, vmin=-1, vmax=1, center=0, cmap='bwr', annot=annot)
        st.pyplot(fig)

    def open(self):
        """ページの表示"""

        # タイトルの表示
        st.title("Random Forest Predictor  ver.1.0")
        st.write("機械学習 (教師あり学習) を手軽に体験できるWebアプリケーションです。")
        st.write("①データの可視化、②データの前処理、③予測 (回帰、分類) ができます。")
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

            with st.expander("目的変数の選択"):
                target = st.selectbox(
                    '目的変数を選択してください',
                    df_uploaded.columns
                )

            with st.expander("基本統計量"):
                st.dataframe(df_uploaded.describe())

            with st.expander("欠損値"):
                st.dataframe(df_uploaded.isnull().sum())

            with st.expander("相関係数"):
                st.dataframe(df_uploaded.corr())
                st.subheader("ヒートマップ")
                heatmap_annot = st.checkbox('ヒートマップに数値を表示')
                self.draw_heatmap(df_uploaded.corr(), heatmap_annot)
                self.draw_pairplot(df_uploaded)
                st.subheader("各説明変数 x 目的変数 の散布図")
                checked_variable = st.selectbox(
                    '説明変数を1つ選択してください:',
                    df_uploaded.select_dtypes(include='number').columns
                )
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.scatter(x=df_uploaded[checked_variable], y=df_uploaded[target])
                plt.xlabel(checked_variable)
                plt.ylabel(target)
                st.pyplot(fig)

            st.subheader("2. データの前処理")
            st.sidebar.header("3. 学習 (フィッティング)")
            # 説明変数と目的変数の設定
            x = df_uploaded.drop(target,axis=1)
            y = df_uploaded[target]

            with st.expander("検証用データの割合"):
                #検証用データの割合を設定
                TEST_SIZE  = st.slider('検証用データの割合を設定', 0.1, 0.5)

            # 学習データと評価データを作成
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=TEST_SIZE,
                random_state=42,
                )

            with st.expander("ランダムフォレストのハイパーパラメータの設定"):
                n_estimators = st.slider("n_estimators", 1, 1000)
                max_depth = st.slider("max_depth", 1, 100)
                criterion = st.radio("criterion", ["mse", "mae", "poisson"])
                bootstrap = st.radio("bootstrap", ["True", "False"])

            if st.button('予測開始'):
                #モデルの学習
                rf_reg = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    criterion=criterion,
                    bootstrap=bootstrap,
                )
                rf_reg.fit(x_train, y_train)

                #予測
                y_pred = rf_reg.predict(x_test)

                #精度評価
                scores = pd.DataFrame({
                    "R2": r2_score(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "MSE": mean_squared_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))},index=["scores"])

                st.sidebar.header("精度評価を表示")
                st.dataframe(scores)

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
