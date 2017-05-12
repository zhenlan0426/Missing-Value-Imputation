# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 14:11:32 2015

@author: zhenlanwang
"""

class card:
    def __init__(self,suit,rank):
        self.suit=suit
        self.rank=rank
    
    suitList = ["Clubs", "Diamonds", "Hearts", "Spades"]
    rankList = ["narf", "Ace", "2", "3", "4", "5", "6", "7","8", "9", "10", "Jack", "Queen", "King"]
    
    def __str__(self):
        return self.rankList[self.rank] + " of " +self.suitList[self.suit]
        
    def __cmp__(self,other):
        if self.suit>other.suit:
            return 1
        elif self.suit<other.suit:
            return -1
        elif self.rank>other.rank:
            return 1
        elif self.rank<other.rank:
            return -1
        else:
            return 0
    
class deck:
    def __init__(self):
        self.cards = []
        for suit in range(4):
            for rank in range(1, 14):
                self.cards.append(card(suit, rank))
    def printDeck(self):
        for i in self.cards:
            print i
    def popCard(self):
        return self.cards.pop()
    def isEmpty(self):
        return (len(self.cards) == 0)
    def deal(self,hands,nCards=999):
        nHands=len(hands)
        for i in range(nCards):
            if self.isEmpty():
                break
            card=self.popCard()
            hand=hands[i % nHands]
            hand.addCard(card)

class hand(deck):
    def __init__(self,name):
        self.cards=[]
        self.name=name
    def addCard(self,card):
        self.cards.append(card)
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
            
