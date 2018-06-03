
# coding: utf-8

# In[ ]:


from selenium import webdriver
from collections import OrderedDict
import pandas as pd
import re
import pickle


# In[1]:


def filter_elos(sample):
    #sample = sample.drop(['Player_White','Player_Black','Tournament','City'],axis = 1) # this is temporary because 

    # count number of row with unr in elo 
    initial = len(sample)
    sample = sample[sample['ELO_Black'].apply(lambda x: str(x).isdigit())]
    sample = sample[sample['ELO_White'].apply(lambda x: str(x).isdigit())]
    resulting = len(sample)
    
    #print(initial-resulting , ' rows dropped')
    return sample

def filter_tournament(sample):
    initial = len(sample)
    Under = sample['Tournament'].apply(lambda x: ('U' in str(x)))
    Women = sample['Tournament'].apply(lambda x: ('women' in str(x).lower()))
    Men = sample['Tournament'].apply(lambda x: ('men' in str(x).lower()))
    
    sample.loc[:,'Tournament'] = 0 # General
    sample.loc[Under,'Tournament'] = 1
    sample.loc[Men,'Tournament'] = 2
    sample.loc[Women,'Tournament'] = 3
                                      
    resulting = len(sample)
    #print(initial-resulting , ' rows dropped')
    return sample

def name_hash(name):
    name = name.lower()
    val = 0
    mod = 1000000007
    
    for i in name:
        val = val * 30 + ord(i) - ord('a')
        val = val % mod
    return val


def filter_player_names(sample):
    
    for i, row in sample.iterrows():
        sample.set_value(i,'Player_White',name_hash(sample.loc[i]['Player_White']))
        sample.set_value(i,'Player_Black',name_hash(sample.loc[i]['Player_Black']))

    return sample





reg = pickle.load(open('rfrModelMiniNEW.sav', 'rb'))

def extractDataFromFen(FEN):
    fenData = []
    fenData.append(len(re.findall('Q',FEN))) #QW
    fenData.append(len(re.findall('q',FEN))) #QB
    fenData.append(len(re.findall('R',FEN))) #RW 
    fenData.append(len(re.findall('r',FEN))) #RB
    fenData.append(len(re.findall('B',FEN))) #BW
    fenData.append(len(re.findall('b',FEN))) #BB
    fenData.append(len(re.findall('N',FEN))) #KW
    fenData.append(len(re.findall('n',FEN))) #KB
    return fenData

def whiteWinning(dw,dd,db):
    return (1/dw)/(1/dw + 1/db + 1/dd)

def blackWinning(dw,dd,db):
    return (1/db)/(1/dw + 1/db + 1/dd)

def matchDraw(dw,dd,db):
    return (1/dd)/(1/dw + 1/db + 1/dd)

browser = webdriver.Firefox() #replace with .Firefox(), or with the browser of your choice
#url = "https://chess24.com/en/watch/live-tournaments/capablanca-memorial-open-2018/8/1/3"
url = "https://chess24.com/en/watch/live-tournaments/limburg-open-2018/3/1/1"
browser.get(url) #navigate to the page

currentGame = browser.find_elements_by_xpath('//span[@class="currentGame"]')

name = [playerName.text for playerName in currentGame][0].split("-")

#fetch black player
player_black = name[0]
#fetch black player elo
elo_black = [eloBlack.text for eloBlack in browser.find_elements_by_xpath('//span[@class="elo black"]')][0]
print("Player Black:"+ player_black + "; EloBlack: "+elo_black)

# fetch white player
player_white = name[1]

#fetch white player elo
elo_white = [eloWhite.text for eloWhite in browser.find_elements_by_xpath('//span[@class="elo white"]')][0]
print("Player White:"+ player_white + "; EloWhite: "+elo_white)

# fetch tournament
tournament = [tournament.text for tournament in browser.find_elements_by_xpath('//h2[@class="title"]')][0]
print("Tournament: "+tournament)

# year
year = 2018
currentMove = ''
flag = 0
analysis = browser.find_elements_by_xpath('//li[@class="tabGamesEngine"]')[0]
analysis.click()
while True:
    if flag==1:
        break
    
    #results = browser.find_elements_by_css_selector('[class*=moveId]')
    results = browser.find_elements_by_xpath('//span[@data-live="Live"]')
    
    x = [result.text for result in results]
    if currentMove!=x[-1] or currentMove == '':
        currentMove = x[-1]
        #print(currentMove)
        if len(currentMove) == 8:
            # white players move
            moveNo = currentMove[:2]
            turn = 1
        else:
            # black players move
            moveNo = currentMove[:2]
            turn = -1
        #print(len(currentMove))
        
        # fetch best evaluatio from Stockfish 7
        moveEval = [moveeval.text for moveeval in browser.find_elements_by_xpath('//th[@class="currentEngineInfo"]')][0]
        print (type(moveEval))
        
        # fen
        fen = [fen.text for fen in browser.find_elements_by_xpath('//span[@class="fen"]')][0]
        fenData = extractDataFromFen(fen)
        
        dataDict = OrderedDict([('Tournament',tournament), ('Year',year), ('Player_White',player_white), ('Player_Black',player_black), ('ELO_White',elo_white), ('ELO_Black',elo_black), ('Turn',turn), ('Move#',moveNo), ('QueenWhite',fenData[0]), ('QueenBlack',fenData[1]), ('RookWhite',fenData[2]), ('RookBlack',fenData[3]), ('BishopWhite',fenData[4]), ('BishopBlack',fenData[5]), ('KnightWhite', fenData[6]), ('KnightBlack',fenData[7]), ('Best_Eval',str(float(moveEval)*float(turn)))])
        
        df = pd.DataFrame(data=dataDict, index=[0])
        #print(df)
        df = filter_player_names(df)
        df = filter_tournament(df)
        df = filter_elos(df)
        
        predicted = reg.predict(df)
        print(predicted[0])
        prediction = predicted[0]
        distance = abs(1-prediction)+0.1
        distance_draw = abs(prediction)+0.1
        distance_black = abs(-1-prediction)+0.1
        
        white = whiteWinning(distance, distance_draw, distance_black)
        black = blackWinning(distance, distance_draw, distance_black)
        draw = matchDraw(distance, distance_draw, distance_black)
        print("-----------------------------------------------------------------------")
        print("Possibility of White Player winning: "+str(white)+" | Possibility of Black player winning: "+str(black)+" | Possibility of a draw: "+str(draw))
        
        
        



# In[ ]:


import pickle
reg = pickle.load(open('rfrModelMini.sav', 'rb'))


# In[ ]:


predicted = reg.predict(modified_df)
print(predicted)


# In[5]:


browser.quit()

