"""
Recommendation
"""

import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from src.utils import prefilter_items


class MainRecommender:
    """Рекомендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать

        # self.data = data
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts()

        # self.sparse_user_item = csr_matrix(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.sparse_user_item = csr_matrix(self.user_item_matrix)

        self.model = self.fit()
        self.own_recommender = self.fit_own_recommender()

    @staticmethod
    def prepare_matrix(data, fake_id=666):
        data = prefilter_items(data)

        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)

        # Взвешиваем по доле продаж в общем объеме
        sales = data.groupby('item_id')['sales_value'].sum().reset_index()
        sales.rename(columns={'sales_value': 'sold'}, inplace=True)
        total_sum = np.sum(sales['sold']) - sales.loc[sales['item_id'] == fake_id, 'sold'].item()
        sales['weight'] = sales['sold'] / total_sum

        for column in user_item_matrix.columns.to_list():
            user_item_matrix.loc[:, column] *= sales.loc[sales['item_id'] == column, 'weight'].item()

        return user_item_matrix

    def prepare_dicts(self):
        """Подготавливает вспомогательные словари"""

        userids = self.user_item_matrix.index.values
        itemids = self.user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    def fit_own_recommender(self):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        # Поменял на K=2, чтобы избежать случаев нулевых рекомендаций
        own_recommender = ItemItemRecommender(K=2, num_threads=4)
        own_recommender.fit(csr_matrix(self.user_item_matrix).T.tocsr())

        return own_recommender

    def fit(self, n_factors=128, regularization=0.05, iterations=10, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())

        return model

    def get_similar_items_recommendation(self, user, n=5, fake_id=666):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        res = list()
        recs = self.get_model_recommendations(user, n)
        for rec in recs:
            similar_items = self.model.similar_items(self.itemid_to_id[rec], N=3)
            item_id = self.id_to_itemid[similar_items[1][0]]
            # Исключаем фиктивный id
            if item_id == fake_id:
                item_id = self.id_to_itemid[similar_items[2][0]]
            res.append(item_id)
            # Позже доработаю, чтобы исключить возможность повторения товаров

        assert len(res) == n, 'Количество рекомендаций != {}'.format(n)
        return res

    def get_similar_users_recommendation(self, user, n=5, fake_id=666):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        # your_code
        res = list()
        similar_userids = self.model.similar_users(self.userid_to_id[user], N=n + 1)
        for userid in similar_userids[1:]:
            user = self.id_to_userid[userid[0]]
            # Добавляем первый рекомендованный товар для каждого из похожих пользователей
            # Позже доработаю, чтобы не было повторов
            res.append(self.get_model_recommendations(user, n=1)[0])

            # Можно брать последний купленный товар с нефиктивным id,
            # но нужно доработать для случая, если все купленные товары с фиктивным id
            # items = self.data.loc[self.data['user_id'] == user, 'item_id'].tolist()
            # j = -1
            # item = items[j]
            # while item == fake_id:
            #     j -= 1
            #     item = items[j]
            # res.append(item)

        assert len(res) == n, 'Количество рекомендаций != {}'.format(n)
        return res

    def get_model_recommendations(self, user, n=5, fake_id=666):
        """Рекомендуем топ-N товаров"""

        res = [self.id_to_itemid[rec[0]] for rec in
               self.model.recommend(userid=self.userid_to_id[user],
                                    user_items=self.sparse_user_item,
                                    N=n,
                                    filter_already_liked_items=False,
                                    filter_items=[self.itemid_to_id[fake_id]],
                                    recalculate_user=True)]
        return res

    def get_own_recommendations(self, user, n=5, fake_id=666):
        """Рекомендуем топ-N товаров"""

        res = [self.id_to_itemid[rec[0]] for rec in
               self.own_recommender.recommend(userid=self.userid_to_id[user],
                                              user_items=self.sparse_user_item,
                                              N=n,
                                              filter_already_liked_items=False,
                                              filter_items=[self.itemid_to_id[fake_id]],
                                              recalculate_user=True)]
        return res
