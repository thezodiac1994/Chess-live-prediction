from flask import Flask, jsonify, request
from flask_cors import CORS
from collections import OrderedDict
app = Flask(__name__)
CORS(app)

import pandas as pd
import pickle

predictions = {}


def filter_elos(sample):
    #sample = sample.drop(['Player_White','Player_Black','Tournament','City'],axis = 1) # this is temporary because 

    # count number of row with unr in elo 
    initial = len(sample)
    sample = sample[sample['ELO_Black'].apply(lambda x: str(x).isdigit())]
    sample = sample[sample['ELO_White'].apply(lambda x: str(x).isdigit())]
    resulting = len(sample)
    
    print(initial-resulting , ' rows dropped')
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
    print(initial-resulting , ' rows dropped')
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

# def readTestInput():
#     df = pd.read_csv("sujit.csv")
#     #print(df)
#     df = filter_player_names(df)
#     df = filter_tournament(df)
#     df = filter_elos(df)
# #    print("----------------Modified------------------")
#     #print(df)
#     modified_df = pd.concat([df.loc[:,'Tournament'], df.loc[:, 'Year':'Eval_Diff']], axis=1)
#     #print(modified_df)
#     return modified_df


reg = pickle.load(open('rfrModelMini.sav', 'rb'))

@app.route('/', methods=['GET'])
def hello_world():
        return jsonify({'message':'It works!'})

@app.route('/test', methods=['GET'])
def predictionApi():
    tournament = request.args.get('tournament')
    year = request.args.get('year')
    playerwhite = request.args.get('playerwhite')
    playerblack = request.args.get('playerblack')
    elowhite = request.args.get('elowhite')
    eloblack  = request.args.get('eloblack')
    turn = request.args.get('turn')
    moveno = request.args.get('moveno')
    queenwhite = request.args.get('queenwhite')
    queenblack = request.args.get('queenblack')
    rookwhite = request.args.get('rookwhite')
    rookblack = request.args.get('rookblack')
    bishopwhite = request.args.get('bishopwhite')
    bishopblack = request.args.get('bishopblack')
    knightwhite = request.args.get('knightwhite')
    knightblack = request.args.get('knightblack')
    besteval = request.args.get('besteval')
    playedeval = request.args.get('playedeval')
    evaldiff = request.args.get('evaldiff')
    dataDict = OrderedDict([('Tournament',tournament), ('Year',year), ('Player_White',playerwhite), ('Player_Black',playerblack), ('ELO_White',elowhite), ('ELO_Black',eloblack), ('Turn',turn), ('Move#',moveno), ('QueenWhite',queenwhite), ('QueenBlack',queenblack), ('RookWhite',rookwhite), ('RookBlack',rookblack), ('BishopWhite',bishopwhite), ('BishopBlack',bishopblack), ('KnightWhite', knightwhite), ('KnightBlack',knightblack), ('Best_Eval',str(float(besteval)*float(turn))), ('Played_Eval',playedeval), ('Eval_Diff',evaldiff)])
    df = pd.DataFrame(data=dataDict, index= [0])
    print(df)
    df = filter_player_names(df)
    df = filter_tournament(df)
    df = filter_elos(df)
    #print(df)
    result = reg.predict(df)
    for i in range(len(result)):
        predictions[i] = result[i]
    print(predictions)
    return jsonify({'predictions':predictions})

if __name__ == '__main__':
        app.run(debug=True, port=8080)

