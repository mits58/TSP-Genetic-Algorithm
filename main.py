# coding:utf-8
import copy
import glob
import math
import os
import shutil
import random as rnd

import matplotlib.pyplot as plt
from PIL import Image
import wandb

rnd.seed(1124)
os.environ['WANDB_MODE'] = 'dryrun'
hyperparameter_defaults = dict(
    pop_num=200,                    # 個体数 -> [100, 300]
    tournament_size=10,             # トーナメントのサイズ -> [5, 100]
    tournament_select_ratio=0.2,    # トーナメントから何割残すか -> [0, 1]
    elite_select_num=1,             # エリートを何個残すか -> [1, 20]
    crossover_prob=50,              # 交叉の確率 -> [0, 100]
    mutation_prob=3,                # 突然変異の確率 -> [0, 100]
)
wandb.init(config=hyperparameter_defaults, project="tsp-genetic-algorithm")
config = wandb.config


def generate_map(num, pop_num):
    """
    引数はnumが地点の数。pop_numが個体数です。
    初期に地点をランダムに作成し、keyを地点番号、座標値をvalueとして辞書を作成しています。
    今回は地点番号の並び順を遺伝子配列としてみなし、アルゴリズムを構築していきます。
    はじめの巡回経路はランダムに作成しています。
    """

    # 範囲指定
    x_range = 200
    y_range = 200

    x_coordinate = [rnd.randint(0, x_range) for _ in range(num)]
    y_coordinate = [rnd.randint(0, y_range) for _ in range(num)]

    coordinate = [[x_coordinate[i], y_coordinate[i]] for i in range(num)]

    # keyが都市の番号、valueが座標値の辞書作成。
    position_info = {}
    for i in range(num):
        position_info[i] = coordinate[i]

    # 初期の巡回順序生成
    select_num = [i for i in range(num)]
    all_route = [rnd.sample(select_num, num) for _ in range(pop_num)]

    return position_info, all_route


def evaluate(position_info, all_route, loop=0):
    """
    ユークリッド距離の総和を評価値として評価関数を設計。
    エリート個体の評価値と、個体全体の評価値の平均を記録。

    show_route()で現世代における最良個体を描画しています。
    """

    evaluate_value = []
    for i in range(len(all_route)):
        temp_evaluate_value = []
        x_coordinate = [position_info[all_route[i][x]][0]
                        for x in range(len(all_route[i]))]
        y_coordinate = [position_info[all_route[i][y]][1]
                        for y in range(len(all_route[i]))]
        for j in range(len(all_route[i])):
            if j == len(all_route[i]) - 1:
                distance = math.hypot((x_coordinate[j] - x_coordinate[0]),
                                      (y_coordinate[j] - y_coordinate[0]))
            else:
                distance = math.hypot((x_coordinate[j] - x_coordinate[j + 1]),
                                      (y_coordinate[j] - y_coordinate[j + 1]))
            temp_evaluate_value.append(distance)
        evaluate_value.append(sum(temp_evaluate_value))

    # エリート個体の評価値と、個体全体の評価値の平均値を記録
    excellent_evaluate_value = min(evaluate_value)
    mean_evaluate_value = sum(evaluate_value) / len(evaluate_value)
    wandb.log({"excellent_evaluate_value": excellent_evaluate_value,
               "mean_evaluate_value": mean_evaluate_value})

    # 一番優秀な個体をmatplotlibで描画、保存
    draw_pop_index = evaluate_value.index(excellent_evaluate_value)
    save_route(position_info, all_route[draw_pop_index],
               int(excellent_evaluate_value), loop=loop)

    return evaluate_value


def selection(all_route, evaluate_value, tournament_select_ratio,
              tournament_size, elite_select_num, ascending=False):
    """
    トーナメント選択、エリート主義を導入。
    トーナメント選択は指定したサイズのトーナメントを作成し、その中で指定数の優良個体を選択。
    エリート主義は指定した上位個体を問答無用で次世代に残す手法。
    """

    select_pop = []
    elite_pop = []
    tournament_select_num = int(tournament_size * tournament_select_ratio)
    # トーナメント選択
    while True:
        select = rnd.sample(evaluate_value, tournament_size)
        select.sort(reverse=ascending)
        for i in range(tournament_select_num):
            value = select[i]
            index = evaluate_value.index(value)
            select_pop.append(all_route[index])

        # 個体数の半数個選択するまで実行
        if len(select_pop) >= len(all_route) / 2:
            break

    # エリート保存
    sort_evaluate_value = copy.deepcopy(evaluate_value)
    sort_evaluate_value.sort(reverse=ascending)
    for i in range(elite_select_num):
        value = sort_evaluate_value[i]
        index = evaluate_value.index(value)
        elite_pop.append(all_route[index])

    return select_pop, elite_pop


def crossover(select_pop, crossover_prob):
    """
    crossover()
    指定した確率によって交叉を実行します。
    ここでは順序交叉と呼ばれる交叉手法を使用しています。
    """

    cross_pop = rnd.sample(select_pop, 2)
    pop_1 = cross_pop[0]
    pop_2 = cross_pop[1]

    check_prob = rnd.randint(0, 100)
    if check_prob <= crossover_prob:

        # 順序交叉
        new_pop_1 = []
        cut_index = rnd.randint(1, len(pop_1) - 2)
        new_pop_1.extend(pop_1[:cut_index])
        for i in range(len(pop_1)):
            if pop_2[i] not in new_pop_1:
                new_pop_1.append(pop_2[i])

        new_pop_2 = []
        new_pop_2.extend(pop_1[cut_index:])
        for i in range(len(pop_1)):
            if pop_2[i] not in new_pop_2:
                new_pop_2.append(pop_2[i])

        return new_pop_1, new_pop_2

    else:
        return pop_1, pop_2


def mutation(pop, mutation_prob):
    """
    指定した確率で突然変異が起こるようにしています。
    今回は地点番号を入れ替える操作を突然変位としています。
    """

    check_prob = rnd.randint(0, 100)

    if check_prob <= mutation_prob:
        select_num = [i for i in range(len(pop))]
        select_index = rnd.sample(select_num, 2)

        a = pop[select_index[0]]
        b = pop[select_index[1]]
        pop[select_index[1]] = a
        pop[select_index[0]] = b

    return pop


def save_route(position_info, route, excellent_evaluate_value, loop=0):
    """
    matplotlibを使用して経路を描画
    """

    x_coordinate = [position_info[route[i]][0] for i in range(len(route))]
    y_coordinate = [position_info[route[i]][1] for i in range(len(route))]
    x_coordinate.append(position_info[route[0]][0])
    y_coordinate.append(position_info[route[0]][1])

    plt.figure(figsize=(4, 4), dpi=100)
    plt.scatter(x_coordinate, y_coordinate)
    plt.plot(x_coordinate, y_coordinate, label=excellent_evaluate_value)
    plt.title("Generation: {}".format(loop))
    plt.legend()
    plt.savefig("img/tsp{0:03}".format(loop),
                bbox_inches="tight", pad_inches=0.05)
    plt.close()


def make_gif(last_value):
    # gif作成
    files = sorted(glob.glob('./img/*.png'))
    images = list(map(lambda file: Image.open(file), files))
    images[0].save(wandb.run.dir + '/out.gif', save_all=True,
                   append_images=images[1:], duration=400, loop=0)
    wandb.log({"transition_best_route": wandb.Video(wandb.run.dir + '/out.gif'),
               "last_best_value": last_value})
    shutil.rmtree('img')
    os.mkdir('img')


def main():
    # 定数
    num_city = 30  # 都市の数
    generation_num = 200  # 世代数

    # imgフォルダのファイルを掃除
    shutil.rmtree('img')
    os.mkdir('img')

    # 初期マップ生成
    position_info, all_route = generate_map(num_city, config.pop_num)

    # 初期評価
    evaluate_value = evaluate(position_info, all_route)

    # 進化操作
    for loop in range(generation_num):
        # 選択
        select_pop, elite_pop = selection(
            all_route, evaluate_value, config.tournament_select_ratio,
            config.tournament_size, config.elite_select_num,
            ascending=False)

        # 選択した個体の中から2個体選択し交叉や突然変異を適用する。
        next_pop = []
        while True:
            # 交叉
            pop_1, pop_2 = crossover(select_pop, config.crossover_prob)
            # 突然変異
            pop_1 = mutation(pop_1, config.mutation_prob)
            pop_2 = mutation(pop_2, config.mutation_prob)

            next_pop.append(pop_1)
            next_pop.append(pop_2)

            if len(next_pop) >= config.pop_num - config.elite_select_num:
                break

        # エリート主義。優良個体を次世代へ継承。
        next_pop.extend(elite_pop)

        # 評価
        evaluate_value = evaluate(position_info, next_pop, loop=loop + 1)

        # 更新
        all_route = next_pop

    # gifの作成と記録
    make_gif(min(evaluate_value))


if __name__ == '__main__':
    main()
