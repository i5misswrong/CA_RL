import matplotlib.pyplot as plt
import Data
import Blcok
import random


class Game():
    def __init__(self):
        print("\033[4;32;40m*******Game stary*********\033[0m")

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
        plt.clf()
        self.drawMap()
        self.drawPed(self.allPeople)
        plt.pause(1)

    def moveFun(self, p, direction):
        isCanMove = True
        if direction == 1:
            if p.x < Data.ROOM_M / 2 + 2 and p.x > Data.ROOM_M / 2 - 2 and p.y + 1 == Data.ROOM_M:
                self.allPeople.remove(p)
            elif p.y + 1 < Data.ROOM_M:
                p.y = p.y + 1
            else:
                isCanMove = False
        elif direction == 2:
            if p.x < Data.ROOM_M / 2 + 2 and p.x > Data.ROOM_M / 2 - 2 and p.x + 1 == Data.ROOM_M:
                pass
            elif p.x + 1 < Data.ROOM_M:
                p.x = p.x + 1
            else:
                isCanMove = False
        elif direction == 3:
            if p.x < Data.ROOM_M / 2 + 2 and p.x > Data.ROOM_M / 2 - 2 and p.y - 1 == Data.ROOM_M:
                pass
            elif p.y - 1 < Data.ROOM_M:
                p.y = p.y - 1
            else:
                isCanMove = False
        elif direction == 4:
            if p.x < Data.ROOM_M / 2 + 2 and p.x > Data.ROOM_M / 2 - 2 and p.y + 1 == Data.ROOM_M:
                pass
            elif p.x - 1 < Data.ROOM_M:
                p.x = p.x - 1
            else:
                isCanMove = False
        return isCanMove

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
        b.x = 10
        b.y = 16
        allPeople.append(b)
        return allPeople

    def getExit(self, p):
        exit_flag = False
        if p.y == Data.ROOM_M and p.x > Data.ROOM_M / 2 - 2 and p.x < Data.ROOM_M / 2 + 2:
            exit_flag = True
        return exit_flag

    def runGame(self, direction):
        self.allPeople = []
        # while Data.FLAG:
        if len(self.allPeople) == 0:
            self.allPeople = self.initSinglePed()
            print("\033[4;32;40mGame restart\033[0m")
            print("\033[0;31;m--------------------------\033[0m")
        for p in self.allPeople:
            self.moveFun(p, direction)
            if self.getExit(p):
                self.allPeople.remove(p)

        self.updateDraw()

    def getMat(self,direction):
        while Data.FLAG:
            self.runGame(1)


g=Game()
g.runGame(1)