# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:43:04 2019

@author: HOME1
"""
import numpy as np
import matplotlib.pyplot as plt

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

def placeboard(move, board, player):
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
nh = 50
ny = 9
layerdims = (nx, nh, ny)


def initparams(layerdims, var=1):
    params = {}
    params['W1'] = np.random.randn(layerdims[1], layerdims[0]) * np.sqrt(var)
    params['W2'] = np.random.randn(layerdims[2], layerdims[1]) * np.sqrt(var)
    
    params['b1'] = np.random.randn(layerdims[1], 1) * np.sqrt(var)
    params['b2'] = np.random.randn(layerdims[2], 1) * np.sqrt(var)
    
    return params

def mutateparams(params, var=1):
    params['W1'] += np.random.randn(*params['W1'].shape) * np.sqrt(var)
    params['W2'] += np.random.randn(*params['W2'].shape) * np.sqrt(var)
    
    params['b1'] += np.random.randn(*params['b1'].shape) * np.sqrt(var)
    params['b2'] += np.random.randn(*params['b2'].shape) * np.sqrt(var)

def mutateparamss(paramss, var=1):
    for params in paramss:
        mutateparams(params, var)

def evolveparamss(paramss, whoparamss, wins, var=1):
    """
    One step of evolution on paramss
    Losers are killed and winners are replicated. Drawers are kept.
    Replicated winners are then mutated.
    """
    if whoparamss == cross:
        winners = (wins == cross)
        losers = (wins == circle)
    elif whoparamss == circle:
        winners = (wins == circle)
        losers = (wins == cross)

    if np.sum(losers) != 0:
        if np.sum(winners) != 0:
            winparamss = paramss[winners]
            for loser in np.nonzero(losers)[0]:
                paramss[loser] = copy.deepcopy(np.random.choice(winparamss))
                mutateparams(paramss[loser], var)
        else:
            mutateparamss(paramss, var)
    
def softmax(z):
    shiftz = z - np.max(z, axis=0) # For numerical stability
    exp = np.exp(shiftz)
    return exp/(np.sum(exp, axis=0))

def forwardprop(x, params):
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']
    
    z1 = W1.dot(x) + b1
    a1 = np.tanh(z1)
    
    z2 = W2.dot(a1) + b2
    y = softmax(z2)
    
    return y

def playstep(params, board):
    """ 
    Play one move for a bot
    input:
        params: params of the bot
        board: board configuration
    output:
        tuple, the desired move position on board for the bot
    """
    x = board.reshape((nx, 1))
    y = forwardprop(x, params)
    argmax = y.argmax()
    return (argmax // 3, argmax % 3)

def playdummystep(board):
    """
    Given the configuration of the board, play a valid move
    Input:
        board: board configuration
    Output:
        tuple of integers, the position of the move
    """
    while True:
        move = (np.random.randint(3), np.random.randint(3))
        if checkvalid(move, board):
            return move

def playround(paramscro, paramscir):
    """ 
    Two bots play against each other and see who wins on one round
    if a bot played an invalid move, it loses
    input: 
        paramscro: params for cross
        paramscir: params for circle
    output:
        integer, the winner
    """
    
    board = newboard()

    while True:
        move = playstep(paramscro, board)
        if checkvalid(move, board):
            placeboard(move, board, cross)
        else:
            win = circle
            break
        if checkwin(board):
            win = cross
            break
        
        if checkfull(board):
            win = blank
            print('draw')
            break
        
        move = playstep(paramscir, board)
        if checkvalid(move, board):
            placeboard(move, board, circle)
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

def playdummyround(params, whoparams):
    """ 
    One bot is played with a dummy bot whose moves are completely random
    They play one round and see who wins
    The dummy bot will not play an invalid move
    If the deterministic bot played an invalid move, it loses
    input:
        params: params of the bot
        whoparams: which player does the params represent
    output: integer, the winner
    """
    board = newboard()
    
    if whoparams == cross:
        while True:
            # Cross turn
            move = playstep(params, board)
            if checkvalid(move, board):
                placeboard(move, board, cross)
            else:
                win = circle
                break
            if checkwin(board):
                win = cross
                break
            
            if checkfull(board):
                win = blank
                print('draw')
                break
            
            # Circle turn
            move = playdummystep(board)
            placeboard(move, board, circle)
            if checkwin(board):
                win = circle
                break
        
    elif whoparams == circle:
        while True:
            # Cross turn
            move = playdummystep(board)
            placeboard(move, board, cross)
            if checkwin(board):
                win = cross
                break
            
            if checkfull(board):
                win = blank
                print('draw')
                break
            
            # Circle turn
            move = playstep(params, board)
            if checkvalid(move, board):
                placeboard(move, board, circle)
            else:
                win = cross
                break
            if checkwin(board):
                win = circle
                break
            
    else:
        print('playdummyround: Error on who the params belong')
        win = None
    
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

def playdummyrounds(paramss, whoparamss):
    wins = []
    for params in paramss:
        wins.append(playdummyround(params, whoparamss))
        
    return np.array(wins, dtype='int')
    
# Create new bots params
numparamscros = 500
numparamscirs = 500
paramscros = np.array([initparams(layerdims) for i in range(numparamscros)])
paramscirs = np.array([initparams(layerdims) for i in range(numparamscirs)])

"""
# Play rounds for cross
paramscir = np.random.choice(paramscirs)
wins = playrounds(paramscros, paramscir, cross)

# Kill losers and replicate winners, keep drawers. Replicated winners are then mutated.
winners = (wins == cross)
losers = (wins == circle)
print('cross: winners:', np.mean(winners), 'losers:', np.mean(losers))
if np.sum(losers) != 0:
    if np.sum(winners) != 0:
        winparamscros = paramscros[winners]
        for loser in np.nonzero(losers)[0]:
            paramscros[loser] = copy.deepcopy(np.random.choice(winparamscros))
            mutateparams(paramscros[loser], var=0.01)
    else:
        mutateparamss(paramscros, var=0.01)

# Play rounds for circle
paramscro = np.random.choice(paramscros)
wins = playrounds(paramscirs, paramscro, circle)

# Kill losers and replicate winners, keep drawers. Replicated winners are then mutated.
winners = (wins == circle)
losers = (wins == cross)
print('circle: winners:', np.mean(winners), 'losers:', np.mean(losers))

if np.sum(losers) != 0:
    if np.sum(winners) != 0:
        winparamscirs = paramscirs[winners]
        for loser in np.nonzero(losers)[0]:
            paramscirs[loser] = copy.deepcopy(np.random.choice(winparamscirs))
            mutateparams(paramscirs[loser], var=0.01)
    else:
        mutateparamss(paramscirs, var=0.01)
"""
winnerss = []
for i in range(5000):
    wins = playdummyrounds(paramscros, cross)
    winners = (wins == cross)
    losers = (wins == circle)
    winnerss.append(np.mean(winners))
    print('winners:', np.mean(winners), 'losers:', np.mean(losers))
    evolveparamss(paramscros, cross, wins, var=0.1)

plt.plot(winnerss)