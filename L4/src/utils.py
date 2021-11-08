"""
Pre- and postfilter items
"""

import numpy as np


def prefilter_items(data, period=53):
    # Оставим топ 5000 популярных товаров
    # popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    # popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    # top_5000 = popularity.sort_values('n_sold', ascending=False).head(5000).item_id.tolist()
    #
    # # Заведем фиктивный item_id для остальных товаров
    # data.loc[~data['item_id'].isin(top_5000), 'item_id'] = 666

    #####

    # Отфильтруем товары, которые продавались в течение последнего года = 53 месяца
    max_week_no = data['week_no'].max()
    last_week_sales = data.groupby('item_id')['week_no'].max().reset_index()
    new = last_week_sales.loc[last_week_sales['week_no'] > max_week_no - period, 'item_id'].tolist()

    fake_id = 666
    data.loc[~data['item_id'].isin(new), 'item_id'] = fake_id

    # Отфильтруем топ 5000 популярных товаров среди тех, у которых средняя цена не очень маленькая и не очень большая
    # Среднюю цену будем вычислять как отношение суммарной sales_value к проданному количеству

    pop_sales = data.groupby('item_id')[['quantity', 'sales_value']].sum().reset_index()
    pop_sales.rename(columns={'quantity': 'n_sold', 'sales_value': 'total_sales'}, inplace=True)
    pop_sales['avg_price'] = np.where(pop_sales['n_sold'] > 0, pop_sales['total_sales'] / pop_sales['n_sold'], 0)
    top_ids = pop_sales[(pop_sales['avg_price'] > 0.5) & (pop_sales['avg_price'] < 5)]. \
        sort_values('n_sold', ascending=False).head(5000).item_id.tolist()

    if fake_id in top_ids:
        top_ids = pop_sales.sort_values('n_sold', ascending=False).head(5001).item_id.tolist()

    data.loc[~data['item_id'].isin(top_ids), 'item_id'] = fake_id

    return data


def postfilter_items(user_id, recommednations):
    pass
