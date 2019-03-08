# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:43:04 2019

@author: HOME1
"""
import numpy as np

import copy

cross = 1
circle = -1
blank = 0

boardshape = (3,3)

def newboard():
    return np.zeros(boardshape)

def checkwin(board):
    #check win for crosses and circles on board
    crosses = np.ones(boardshape) * (cross == board)
    circles = np.ones(boardshape) * (circle == board)
    if 3 in crosses.sum(axis=0) or 3 in crosses.sum(axis=1) or crosses.trace() == 3 or np.fliplr(crosses).trace() == 3:
        return cross
    if 3 in circles.sum(axis=0) or 3 in circles.sum(axis=1) or circles.trace() == 3 or np.fliplr(circles).trace() == 3:
        return circle
    return blank

def checkfull(board):
    if blank in board:
        return False
    else:
        return True
    
def checkvalid(move, board):
    if blank == board[move[0], move[1]]:
        return True
    else:
        return False

def placeboard(move, player, board):
    if checkvalid(move, board):
        board[move[0], move[1]] = player
    else:
        print('placeboard: move invalid')
    
"""
board = newboard()
print(board)
while not checkwin(board) and not checkfull(board):
    while(True):
        move = input('y x player: ').split()
        move = [int(num) for num in move]
        if checkvalid(move[0:2], board):
            break
    board[move[0], move[1]] = move[2]
    print(board)
"""

nx = 9
nh = 10
ny = 9
layerdims = (nx, nh, ny)


def initparams(layerdims, var = 1):
    params = {}
    params['Wax'] = np.random.randn(layerdims[1], layerdims[0]) * np.sqrt(var)
    params['Waa'] = np.random.randn(layerdims[1], layerdims[1]) * np.sqrt(var)
    params['Wy'] = np.random.randn(layerdims[2], layerdims[1]) * np.sqrt(var)
    
    params['ba'] = np.random.randn(layerdims[1], 1) * np.sqrt(var)
    params['by'] = np.random.randn(layerdims[2], 1) * np.sqrt(var)
    
    return params

def mutateparams(params, var = 1):
    params['Wax'] += np.random.randn(*params['Wax'].shape) * np.sqrt(var)
    params['Waa'] += np.random.randn(*params['Waa'].shape) * np.sqrt(var)
    params['Wy'] += np.random.randn(*params['Wy'].shape) * np.sqrt(var)
    
    params['ba'] += np.random.randn(*params['ba'].shape) * np.sqrt(var)
    params['by'] += np.random.randn(*params['by'].shape) * np.sqrt(var)

def softmax(z):
    exp = np.exp(z)
    return exp/(np.sum(exp, axis=0))
    
def forwardprop(x, a, params):
    Wax = params['Wax']
    Waa = params['Waa']
    Wy = params['Wy']
    ba = params['ba']
    by = params['by']
    
    za = Wax.dot(x) + Waa.dot(a) + ba
    a = np.tanh(za)
    
    zy = Wy.dot(a) + by
    y = softmax(zy)
    
    return y, a



params2 = initparams(layerdims)
move1 = np.zeros(2, dtype='int')
move2 = np.zeros(2, dtype='int')


numparams1 = 10
params1s = np.array([initparams(layerdims) for i in range(numparams1)])

wins1 = np.zeros(numparams1, dtype='int')
for ind, params1 in enumerate(params1s):
    board = newboard()
    x = board.reshape((nx, 1))
    a1 = np.zeros((nh, 1))
    a2 = np.zeros((nh, 1))
    
    while not checkfull(board):
        y1, a1 = forwardprop(x, a1, params1)
        arg1 = y1.ravel().argsort()[::-1]
        for i in range(9):
            move1[0] = arg1[i] // 3
            move1[1] = arg1[i] % 3
            if checkvalid(move1, board):
                placeboard(move1 , cross, board)
                break
        if checkwin(board):
            #print('X wins')
            wins1[ind] = cross
            break
        
        
        y2, a2 = forwardprop(x, a2, params2)
        arg2 = y2.ravel().argsort()[::-1]
        for i in range(9):
            move2[0] = arg2[i] // 3
            move2[1] = arg2[i] % 3
            if checkvalid(move2, board):
                placeboard(move2 , circle, board)
                break
        if checkwin(board):
            #print('O wins')
            wins1[ind] = circle
            break
    if checkwin(board) == blank:
        #print('Draw')
        pass

print(np.mean(wins1))


winparams1s = params1s[wins1 == cross]

for ind in np.nonzero(wins1 != cross)[0]:
    params1s[ind] = copy.deepcopy(np.random.choice(winparams1s))





