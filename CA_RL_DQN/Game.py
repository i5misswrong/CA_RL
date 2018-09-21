import matplotlib.pyplot as plt
import Data
import Blcok
import random
import numpy as np


class Game():
    def __init__(self):
        '''
        初始化：
        action_space:动作空间
        n_action:动作空间长度
        getExit:是否到达出口
        isNearExit:是否在出口附近
        counterWall:是否碰到墙壁
        '''
        print("\033[4;32;40m*******Game start*********\033[0m")
        self.action_space = [0, 1, 2, 3]
        self.n_actions = len(self.action_space)
        self.getExit = False
        self.isNearExit = False
        self.counterWall = False

    def drawPed(self, allPeople):
        '''
        绘制行人 传入行人数组，获取xy坐标，绘制
        :param allPeople: 行人数组
        :return: null
        '''
        coord_x = []
        coord_y = []
        for p in allPeople:
            coord_x.append(p.x)
            coord_y.append(p.y)
        plt.scatter(coord_x, coord_y, c='r', marker='o')

    def drawMap(self):
        '''
        绘制地图，由4根线组成
        :return: null
        '''
        # around wall
        plt.plot([0, Data.ROOM_M], [0, 0], 'k-')  # down
        plt.plot([Data.ROOM_M / 2 - 2, 0], [Data.ROOM_M, Data.ROOM_M], 'k-')  # up
        plt.plot([Data.ROOM_M, Data.ROOM_M / 2 + 2], [Data.ROOM_M, Data.ROOM_M], 'k-')  # up
        plt.plot([0, 0], [0, Data.ROOM_M], 'k-')  # left
        plt.plot([Data.ROOM_M, Data.ROOM_M], [0, Data.ROOM_M], 'k-')  # right
        # exit
        plt.plot([Data.ROOM_M / 2 - 2, Data.ROOM_M / 2 + 2], [Data.ROOM_M, Data.ROOM_M], 'b--')

    def updateDraw(self):
        '''
        重绘，更新帧
        :return: null
        '''
        # plt.figure(num=1,figsize=(4,4))
        plt.clf()
        self.drawMap()  # 绘制地图
        self.drawPed(self.allPeople)  # 绘制行人
        plt.pause(1e-4)  # 暂停0.1s

    def moveFun(self, p, direction):
        '''
        行人移动方法
        :param p: 当前行人
        :param direction:传入移动方向
        :return: null
        '''
        bo = self.isCanMove(p, direction)  # 判断行人能否移动
        isCouldMove = bo[0]  # 行人下一点是否移动到墙壁
        isGetExit = bo[1]  # 行人是否到达出口
        if isGetExit:  # 如果到达出口
            self.allPeople.remove(p)  # 移除该行人
            self.getExit = True  # 到达出口flag设为True
        elif isCouldMove:  # 如果行人能移动
            if direction == 0:  # 上
                p.y = p.y + 1
            elif direction == 1:  # 右
                p.x = p.x + 1
            elif direction == 2:  # 下
                p.y = p.y - 1
            elif direction == 3:  # 左
                p.x = p.x - 1

    def isCanMove(self, p, direction):
        '''
        行人能否移动，判断下一点坐标是否为墙壁
        :param p: 当前行人
        :param direction:方向
        :return: 列表 [行人能否移动，行人是否到达出口]
        '''
        isCouldMove = True  # 初始化俩参数
        isGetExit = False
        p_next_position = self.next_position(p, direction)  # 根据移动方向获取行人下一点坐标
        p_x_ = p_next_position[0]  # 获取x坐标
        p_y_ = p_next_position[1]  # 获取y坐标

        if p_x_ < Data.ROOM_M / 2 + 2 and p_x_ > Data.ROOM_M / 2 - 2 and p_y_ == Data.ROOM_M:
            # 如果行人位于出口范围内
            print("the pedestrian in exit")
            isGetExit = True  # 设置flag
        if p_x_ < Data.ROOM_M / 2 + 2 and p_x_ > Data.ROOM_M / 2 - 2 and p_y_ == Data.ROOM_M - 1:
            # 如果行人在出口下面一格 （接近出口）
            self.isNearExit = True  # 设置flag
        if p_x_ <= 0 or p_x_ >= Data.ROOM_M or p_y_ <= 0 or p_y_ >= Data.ROOM_M:
            # 如果行人下一点在疏散区域外
            # print("the pedestrian go out evacuation area")
            isCouldMove = False  # 设置为不可移动
            self.counterWall = True  # 设置为撞墙了
        bo = []  # 返回参数
        bo.append(isCouldMove)  # 添加参数
        bo.append(isGetExit)
        return bo

    def next_position(self, p, direction):
        '''
        根据方向获取行人下一点坐标
        :param p: 当前行人
        :param direction: 方向
        :return:
        '''
        p_x_ = 0  # 初始化x y坐标
        p_y_ = 0
        if direction == 0:  # 上
            p_x_ = p.x
            p_y_ = p.y + 1
        elif direction == 1:  # 右
            p_x_ = p.x + 1
            p_y_ = p.y
        elif direction == 2:  # 下
            p_x_ = p.x
            p_y_ = p.y - 1
        elif direction == 3:  # 左
            p_x_ = p.x - 1
            p_y_ = p.y
        p_next_postition = []  # 行人位置列表
        p_next_postition.append(p_x_)  # x坐标
        p_next_postition.append(p_y_)  # y坐标
        return p_next_postition

    def initRandomPed(self):
        '''
        随机初始化行人
        :return: 包含所有行人的列表
        '''
        allBlock = []  # 疏散空间所有xy坐标
        allPeople = []  # 所有行人
        # 获取疏散空间所有xy坐标
        for i in range(1, Data.ROOM_M):
            for j in range(1, Data.ROOM_M):
                b = Blcok.Block()  # 初始化block
                b.x = i  # 赋予坐标
                b.y = j

        random.shuffle(allBlock)  # 随机排序
        allPeople = allBlock[:Data.PEOPLE_NUMBER]  # 取出前N个行人
        return allPeople

    def initSinglePed(self):
        '''
        初始化单个行人
        :return: 单个行人  列表
        '''
        allPeople = []
        b = Blcok.Block()
        b.x = 3
        b.y = 3
        allPeople.append(b)
        return allPeople

    # def runGame(self, direction):
    #     self.allPeople = []
    #     # while Data.FLAG:
    #     if len(self.allPeople) == 0:
    #         self.allPeople = self.initSinglePed()
    #         print("\033[4;32;40mGame restart\033[0m")
    #         print("\033[0;31;m--------------------------\033[0m")
    #     for p in self.allPeople:
    #         self.moveFun(p, direction)
    #         if self.getExit(p):
    #             self.allPeople.remove(p)
    #
    #     self.updateDraw()

    # def getMat(self, direction):
    #     while Data.FLAG:
    #         self.runGame(1)

    # todo reset and render 返回一个二维矩阵
    # todo 实时更新矩阵

    def creatMat(self, p_x, p_y):
        mat = np.zeros((Data.ROOM_M, Data.ROOM_M))
        # matplotlib 和 numpy 坐标轴方向不一样
        mat[p_x, p_y] = 100
        return mat

    def reset(self):
        '''
        重置环境
        :return: 行人坐标
        '''
        print("\033[4;36;40m------reset------\033[0m")
        self.getExit = False
        self.allPeople = self.initSinglePed()  # 初始化行人
        self.updateDraw()  # 更新 帧
        p = self.allPeople[0]  # 获取行人
        # p_o = []
        # p_o.append(p.x)
        # p_o.append(p.y)
        mat = self.creatMat(p.x, p.y)
        return mat  # 返回行人坐标

    def step(self, action):  # 单步运行
        '''
        单步运行
        :param action:当前动作==当前方向
        :return: 下一步状态，期望值，终止符
        '''
        reward = -0.1  # 期望值
        done = False  # 终止符
        s_ = self.creatMat(1, 1)  # 下一步状态
        for p in self.allPeople:
            # if p.x < Data.ROOM_M / 2 + 2 and p.x > Data.ROOM_M / 2 - 2 and p.y>= Data.ROOM_M:
            #     print('in step, pedestrian really position is get out exit')
            # else:
            if self.moveFun(p, action):  # 行人移动
                pass
            if self.getExit:  # 如果到达出口
                reward = 10  # 期望=10
                done = True  # 终止符=True
                # s_ = "terminal"  #
            else:
                # if self.counterWall:  # 如果到达墙壁
                #     reward = -1  # 期望=-1
                # if self.isNearExit:  # 如果在出口附近
                #     reward = 0.5  # 期望=0.5
                pass
            # print(p.x, p.y, '----')
            s_ = self.creatMat(p.x, p.y)
            self.counterWall = False  # 重置参数
            self.isNearExit = False
        return s_, reward, done

    def render(self):  # 更新环境
        self.updateDraw()


# g = Game()
# for i in range(10):
#     g.reset()
#     flag_done = False
#     while not flag_done:
#         s_, reward, done = g.step(0)
#         g.render()
#         print(s_)
#         print('---reward=', reward, '---done=', done)
#         flag_done = done
#
#     print('----------------')
