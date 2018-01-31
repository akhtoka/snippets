import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from IPython.display import display

#%matplotlib inline
#plt.style.use('ggplot')
#plt.rcParams['font.family'] = 'IPAexGothic'



class AnalysisSnippets():

    def hml_kmeans(self, clst_list, n_clst):
        # HML用のkmeansクラスタリング
        # クラスタ結果の表示、クラスタ番号を平均の高い順に変更する

        RANDOM_STATE = 123
        vector_array = np.array(clst_list)
        ROWS_NUM = len(vector_array)
        vector_array = vector_array.reshape(ROWS_NUM, 1)

        clusters = KMeans(n_clusters=n_clst, random_state=RANDOM_STATE).fit_predict(vector_array)

        # 元のリストとクラスタでpd.DataFrameの作成
        clst_df = pd.DataFrame({"var": clst_list, "cluster": clusters})

        # クラスタ毎の統計量を確認し、デフォルトではmeanでクラスタ番号を昇順に
        clst_result = clst_df.groupby(["cluster"])["var"].agg(["mean", "count", "min", "max", "median"]).reset_index()
        clst_result['rank_by_mean'] = clst_result["mean"].rank(ascending=False).astype("int")
        display(clst_result)
        convert_rules = dict(clst_result.set_index("cluster").rank_by_mean)

        return list(clst_df["cluster"].map(convert_rules))


    def print_cmx(self, y_true, y_pred, labels):

        # めっちゃ綺麗な混合行列を出力する関数
        # ヒートマップ機能付き

        cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
        print(cmx_data)
        df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

        plt.figure(figsize=(6, 4.5))
        plt.rcParams['font.size'] = 15
        sn.heatmap(df_cmx, cmap='Reds', annot=True, fmt="d", xticklabels=True, yticklabels=True)
        # plt.savefig('heatmap.png')
        ax = plt.gca()
        ax.xaxis.set_ticks_position("top")  # x軸目盛りは軸の上、bottomで下になる
        ax.set_yticklabels(reversed(labels), rotation=0)
        ax.set_xlabel("predicted label")
        ax.set_ylabel("real label")
        plt.show()


    def RF_classifier(self, df, y_column, feature_columns, test_rate):

        # クラス分類用ランダムフォレスト
        # 混合行列や重要度の高い変数を可視化する

        # 説明変数、目的変数の作成
        X = df.loc[:, feature_columns].values
        Y = df.loc[:, y_column].values

        # 学習用、検証用データに分割
        (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=test_rate, random_state=123)

        '''
        # モデル構築、パラメータはデフォルト
        parameters = {
            'n_estimators'      : [5, 10, 20, 30, 50],
            'max_features'      : [3, 5, 10, 15, 20],
            'random_state'      : [0],
            'n_jobs'            : [2],
            'min_samples_split' : [3, 5, 10, 15, 20, 25, 30],
            'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 50, 100]
        }

        clf = GridSearchCV(RandomForestClassifier(), parameters)
        clf.fit(X_train, Y_train)
        print(clf.best_estimator_)'''

        model = RandomForestClassifier(n_estimators=20, max_depth=4, max_features=None, bootstrap=True)

        print(model.get_params())
        model.fit(X_train, Y_train)

        # 正解率
        print("正解率 : " + str(model.score(X_test, Y_test) * 100) + "%")
        print("訓練データの正解率 : " + str(model.score(X_train, Y_train) * 100) + "%")

        # confusion matrix　を確認する
        print("confusion matrix")
        prediction = model.predict(X_test)
        labels = list(set(Y))
        self.print_cmx(Y_test, prediction, labels)

        # 効いてる変数を順に並べる
        importances = pd.DataFrame(
            {
                'variable': feature_columns,
                'importance': model.feature_importances_
            }
        ).sort_values('importance', ascending=False).reset_index(drop=True)
        display(importances)

        IMP = importances.copy()
        plt.figure(figsize=(5, 7))
        plt.plot(IMP.importance, sorted([i + 1 for i in range(IMP.shape[0])], reverse=True), 'o-')
        plt.yticks(sorted([i + 1 for i in range(IMP.shape[0])], reverse=True), IMP.variable)
        plt.xlabel('importance')
        # plt.xlabel('重要度')
        plt.show()

        return importances


