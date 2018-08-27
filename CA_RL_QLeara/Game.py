import matplotlib.pyplot as plt
import Data
import Blcok
import random


class Game():
    def __init__(self):
        print("\033[4;32;40m*******Game start*********\033[0m")
        self.action_space = [0, 1, 2, 3]
        self.n_actions = len(self.action_space)
        self.getExit = False
        self.isNearExit=False
        self.counterWall = False

    def drawPed(self, allPeople):
        coord_x = []
        coord_y = []
        for p in allPeople:
            coord_x.append(p.x)
            coord_y.append(p.y)
        plt.scatter(coord_x, coord_y, c='r', marker='o')

    def drawMap(self):
        # around wall
        plt.plot([0, Data.ROOM_M], [0, 0], 'k-')  # down
        plt.plot([Data.ROOM_M / 2 - 2, 0], [Data.ROOM_M, Data.ROOM_M], 'k-')  # up
        plt.plot([Data.ROOM_M, Data.ROOM_M / 2 + 2], [Data.ROOM_M, Data.ROOM_M], 'k-')  # up
        plt.plot([0, 0], [0, Data.ROOM_M], 'k-')  # left
        plt.plot([Data.ROOM_M, Data.ROOM_M], [0, Data.ROOM_M], 'k-')  # right
        # exit
        plt.plot([Data.ROOM_M / 2 - 2, Data.ROOM_M / 2 + 2], [Data.ROOM_M, Data.ROOM_M], 'b--')

    def updateDraw(self):

        # plt.figure(num=1,figsize=(4,4))
        plt.clf()
        self.drawMap()
        self.drawPed(self.allPeople)
        plt.pause(0.01)

    def moveFun(self, p, direction):
        bo = self.isCanMove(p, direction)
        isCouldMove = bo[0]
        isGetExit = bo[1]
        if isGetExit:
            self.allPeople.remove(p)
            self.getExit = True
        elif isCouldMove:
            if direction == 0:
                p.y = p.y + 1
            elif direction == 1:
                p.x = p.x + 1
            elif direction == 2:
                p.y = p.y - 1
            elif direction == 3:
                p.x = p.x - 1

    def isCanMove(self, p, direction):
        isCouldMove = True
        isGetExit = False
        p_next_position = self.next_position(p, direction)
        p_x_ = p_next_position[0]
        p_y_ = p_next_position[1]

        if p_x_ < Data.ROOM_M / 2 + 2 and p_x_ > Data.ROOM_M / 2 - 2 and p_y_ == Data.ROOM_M:
            print("the pedestrian in exit")
            isGetExit = True
        if p_x_ < Data.ROOM_M / 2 + 2 and p_x_ > Data.ROOM_M / 2 - 2 and p_y_ == Data.ROOM_M-1:
            self.isNearExit=True
        if p_x_ <= 0 or p_x_ >= Data.ROOM_M or p_y_ <= 0 or p_y_ >= Data.ROOM_M:
            # print("the pedestrian go out evacuation area")
            isCouldMove = False
            self.counterWall = True
        bo = []
        bo.append(isCouldMove)
        bo.append(isGetExit)
        return bo

    def next_position(self, p, direction):
        p_x_ = 0
        p_y_ = 0
        if direction == 0:
            p_x_ = p.x
            p_y_ = p.y + 1
        elif direction == 1:
            p_x_ = p.x + 1
            p_y_ = p.y
        elif direction == 2:
            p_x_ = p.x
            p_y_ = p.y - 1
        elif direction == 3:
            p_x_ = p.x - 1
            p_y_ = p.y
        p_next_postition = []
        p_next_postition.append(p_x_)
        p_next_postition.append(p_y_)
        return p_next_postition

    def initRandomPed(self):
        allBlock = []
        allPeople = []
        for i in range(1, Data.ROOM_M):
            for j in range(1, Data.ROOM_M):
                b = Blcok.Block()
                b.x = i
                b.y = j

        random.shuffle(allBlock)
        allPeople = allBlock[:Data.PEOPLE_NUMBER]
        return allPeople

    def initSinglePed(self):
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

    def reset(self):  # 重置环境
        print("\033[4;36;40m------reset------\033[0m")
        self.getExit = False
        self.allPeople = self.initSinglePed()
        self.updateDraw()
        p = self.allPeople[0]
        p_o = []
        p_o.append(p.x)
        p_o.append(p.y)
        return p_o

    def step(self, action):  # 单步运行
        reward = 0
        done = False
        s_ = ""
        for p in self.allPeople:
            if self.moveFun(p, action):
                pass
            if self.getExit:
                reward = 10
                done = True
                s_ = "terminal"
            if self.counterWall:
                reward=-1
            if self.isNearExit:
                reward=0.5
            self.counterWall=False
            self.isNearExit=False
        return s_, reward, done

    def render(self):  # 更新环境
        self.updateDraw()

# g = Game()
# g.reset()
# for i in range(10):
#     g.step(2)
#     g.render()
