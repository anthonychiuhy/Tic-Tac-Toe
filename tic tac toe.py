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
    return np.zeros(boardshape, dtype='int')

def checkwin(board):
    #check win for crosses and circles on board
    crosses = np.ones(boardshape) * (cross == board)
    circles = np.ones(boardshape) * (circle == board)
    if 3 in crosses.sum(axis=0) or 3 in crosses.sum(axis=1) or crosses.trace() == 3 or np.fliplr(crosses).trace() == 3:
        return True
    if 3 in circles.sum(axis=0) or 3 in circles.sum(axis=1) or circles.trace() == 3 or np.fliplr(circles).trace() == 3:
        return True
    return False

def checkfull(board):
    if blank in board:
        return False
    else:
        return True
    
def checkvalid(move, board):
    if blank == board[move]:
        return True
    else:
        return False

def placeboard(move, player, board):
    if checkvalid(move, board):
        board[move] = player
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
nh = 100
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

def mutateparamss(paramss, var = 1):
    for params in paramss:
        mutateparams(params, var)

def softmax(z):
    shiftz = z - np.max(z, axis=0) # For numerical stability
    exp = np.exp(shiftz)
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



def playround(paramscro, paramscir):
    """ 
    Two bots play against each other and see who wins on one round
    if a bot played an invalid move, it loses
    input: params of two bots, paramscro will move first
    output: integer, the winner
    """
    
    board = newboard()
    x = board.reshape((nx, 1))
    a1 = np.zeros((nh, 1))
    a2 = np.zeros((nh, 1))
    
    move1 = np.zeros(2, dtype='int')
    move2 = np.zeros(2, dtype='int')
    
    while True:
        y1, a1 = forwardprop(x, a1, paramscro)
        argmax1 = y1.argmax()
        move1 = (argmax1 // 3, argmax1 % 3)
        if checkvalid(move1, board):
            placeboard(move1, cross, board)
        else:
            win = circle
            break
        if checkwin(board):
            win = cross
            break
        
        if checkfull(board):
            win = blank
            break
        
        y2, a2 = forwardprop(x, a2, paramscir)
        argmax2 = y2.argmax()
        move2 = (argmax2 // 3, argmax2 % 3)
        if checkvalid(move2, board):
            placeboard(move2 , circle, board)
        else:
            win = cross
            break
        if checkwin(board):
            win = circle
            break
        
        if checkfull(board):
            win = blank
            print('playround: Impossible to activate this')
            break
    
    return win


def playrounds(paramss, paramsfixed, whoparamss):
    """ 
    Two bots play against each other on multiple rounds.
    multiple params are passed for one player and one fixed params are passed for the other player
    input:
        paramss: consists of multiple params for a single bot
        paramsfixed: single params for the other bot
        whoparamss: signifies which player the paramss belong to
    output: numpy array of winners on every round
    """
    wins = []
    if whoparamss == cross:
        for params in paramss:
            wins.append(playround(params, paramsfixed))
    elif whoparamss == circle:
        for params in paramss:
            wins.append(playround(paramsfixed, params))
    else:
        print('playrounds: Error on who the params belong')
        
    return np.array(wins, dtype='int')
    

# Create new bots params
numparamscros = 100
numparamscirs = 100
paramscros = np.array([initparams(layerdims) for i in range(numparamscros)])
paramscirs = np.array([initparams(layerdims) for i in range(numparamscirs)])

for j in range(50):
    winss = []
    losess = []
    
    for i in range(100):
        # Play rounds
        paramscir = np.random.choice(paramscirs)
        wins = playrounds(paramscros, paramscir, cross)
        
        # Kill losers and replicate winners, keep drawers. Replicated winners are then mutated.
        winners = (wins == cross)
        losers = (wins == circle)
        print('winners:', np.mean(winners), 'losers:', np.mean(losers))
        if np.sum(losers) != 0:
            if np.sum(winners) != 0:
                winparamscros = paramscros[winners]
                for loser in np.nonzero(losers)[0]:
                    paramscros[loser] = copy.deepcopy(np.random.choice(winparamscros))
                    mutateparams(paramscros[loser])
            else:
                mutateparamss(paramscros)
        
        winss.append(np.mean(winners))
        losess.append(np.mean(losers))
    
    print(np.mean(winss), np.mean(losess))
    print('###########################################################')
    
    
    
    winss = []
    losess = []
    
    for i in range(100):
        # Play rounds
        paramscro = np.random.choice(paramscros)
        wins = playrounds(paramscirs, paramscro, circle)
        
        # Kill losers and replicate winners, keep drawers. Replicated winners are then mutated.
        winners = (wins == circle)
        losers = (wins == cross)
        print('winners:', np.mean(winners), 'losers:', np.mean(losers))
    
        if np.sum(losers) != 0:
            if np.sum(winners) != 0:
                winparamscirs = paramscirs[winners]
                for loser in np.nonzero(losers)[0]:
                    paramscirs[loser] = copy.deepcopy(np.random.choice(winparamscirs))
                    mutateparams(paramscirs[loser])
            else:
                mutateparamss(paramscirs)
        
        winss.append(np.mean(winners))
        losess.append(np.mean(losers))
    
    print(np.mean(winss), np.mean(losess))
