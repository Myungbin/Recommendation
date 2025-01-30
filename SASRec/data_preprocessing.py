from collections import defaultdict

import pandas as pd


def data_partition():

    columns = ["userid", "movieid", "rating", "timestamp"]
    data = pd.read_table("./data/ml-1m/ratings.dat", names=columns, sep="::", encoding="latin1", engine="python")
    raw_data = data[["userid", "movieid"]]

    user_num = raw_data["userid"].max()
    item_num = raw_data["movieid"].max()
    user_item_dict = data.groupby("userid")["movieid"].agg(list).to_dict()

    user_train, user_valid, user_test = defaultdict(list), defaultdict(list), defaultdict(list)

    for user, items in user_item_dict.items():
        if len(items) < 5:
            user_train[user] = items
            print(user_valid[user])
        else:
            user_train[user] = items[:-2]
            user_valid[user].append(items[-2])
            user_test[user].append(items[-1])

    return [user_train, user_valid, user_test, user_num, item_num]


def data_partition():

    columns = ["userid", "movieid", "rating", "timestamp"]
    data = pd.read_table("./data/ml-1m/ratings.dat", names=columns, sep="::", encoding="latin1", engine="python")
    raw_data = data[["userid", "movieid"]]

    user_num = raw_data["userid"].max()
    item_num = raw_data["movieid"].max()
    user_item_dict = data.groupby("userid")["movieid"].agg(list).to_dict()

    user_train, user_valid, user_test = {}, {}, {}

    for user in user_item_dict:
        feedback_num = len(user_item_dict[user])
        if feedback_num < 5:
            user_train[user] = user_item_dict[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = user_item_dict[user][:-2]
            user_valid[user] = []
            user_valid[user].append(user_item_dict[user][-2])
            user_test[user] = []
            user_test[user].append(user_item_dict[user][-1])

    return [user_train, user_valid, user_test, user_num, item_num]
