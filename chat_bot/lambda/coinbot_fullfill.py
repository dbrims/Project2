from botocore.vendored import requests
import ccxt
import os 
from io import StringIO
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import boto3
import json



s3 = boto3.client('s3')

def parse_float(n):
    """
    Securely converts a non-numeric value to float.
    """
    try:
        return float(n)
    except ValueError:
        return "error"
    

    
def risk_assay(bday, income, amount, risk, retire):
    '''from the clients inputs we are going to assign them a risk score high risk=2, moderate risk=2, low risk=1'''

    rlst=[]
    birth_date = datetime.strptime(bday, "%Y-%m-%d")
    age = relativedelta(datetime.now(), birth_date).years
    working_age=retire-age
    income_ratio=income/amount
    
    '''we are going to use their age and retirement age to create a risk value, closer to retirement, the higher the risk'''

    if working_age<17:
        rlst.append(3)
    elif working_age>=17 and working_age<34:
        rlst.append(2)
    else:
        rlst.append(1)
        
    '''we will use their income and amount to invest.  If the investment amount is a sizable chunk of their annual income
    we will say it is higher risk'''
    if income_ratio<2:
        rlst.append(3)
    elif income_ratio>=2 and income_ratio<5:
        rlst.append(2)
    else:
        rlst.append(1)
        
    '''we will use their explicitly stated risk as an input also,  if they entered giberish, we will ask them for clarification'''
    if risk.lower()=='high':
        rlist.append(3)
    elif risk.lower()=='high':
        rlst.append(2)
    elif risk.lower()=='low':
        rlst.append(1)
        
    ave_risk = round(sum(rlst) / len(rlst),0)
    
    return(ave_risk)



def get_coins():
    
    '''here we read the coins csv file from the E3 drive and pass it back'''
    bucket = 'ft-project-2'
    key = 'Classification.csv'
    response = s3.get_object(Bucket=bucket, Key=key)
    csv_file=response['Body']
    df=pd.read_csv(csv_file)


    return(df)



def get_tickers(ave_risk, coin_df):
    '''we are saying cryto class have a risk associated with them and based 
    on the risk metric for the client we are assigning them one of those risk 
    portfolios'''
    tickers=[]
    if ave_risk==1:
        symbols=coin_df['symbol'].loc[(coin_df['class']==2)|(coin_df['class']==0)].to_list()
    elif ave_risk==2:
        symbols=coin_df['symbol'].to_list()
    elif ave_risk==3:
        symbols=coin_df['symbol'].loc[(coin_df['class']==1)|(coin_df['class']==0)].to_list()
 
    return(symbols)



def get_portfolio(symbols):
    i=0
    exchange = ccxt.kraken({
    'apiKey': KRAKEN_API_KEY,
    'secret': KRAKEN_SECRET_API_KEY,
    })
    since = exchange.parse8601('2018-01-01T00:00:00z')

    for symbol in symbols:
        if i==0:
            try:
                data = exchange.fetchOHLCV(symbol,'1d', since, params = {})
                header = ['Date', 'Open', 'High', 'Low', symbol, 'Volume']
                ticker_df = pd.DataFrame(data, columns=header)
                ticker_df['Date']=pd.to_datetime(ticker_df.Date/1000, unit='s')
                ticker_df.set_index('Date', inplace=True)
                ticker_df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)
                i+=1
            except:
                continue
        else:
            try:
                data = exchange.fetchOHLCV(symbol,'1d', since, params = {})
                header = ['Date', 'Open', 'High', 'Low', symbol, 'Volume']
                df = pd.DataFrame(data, columns=header)
                df['Date']=pd.to_datetime(df.Date/1000, unit='s')
                df.set_index('Date', inplace=True)
                df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)
                ticker_df=pd.concat([ticker_df, df], axis=1, join='inner')
            except:
                continue

    return(ticker_df)


    
    
def make_data_df(coin_df, tickers_df, asset_returns, symbols):
    tickers=tickers_df.columns.tolist()
    data_df=pd.DataFrame()
    data_df['tickers']=tickers
    data_df['returns']=asset_returns
    data_df.set_index('tickers', inplace=True)
    data_df['mrkcap']=0
    data_df['price']=0

    clean_tickers, data_df=fill_data_df(coin_df, data_df, tickers_df, tickers)
    tickers_df=prune_tickers_df(tickers,clean_tickers, tickers_df)
    return(clean_tickers, data_df, tickers_df)


def fill_data_df(coin_df,data_df, tickers_df, tickers):
    last_price=tickers_df.iloc[-1,:]
    for ticker in tickers:
        data_df.loc[ticker,'mrkcap']=coin_df.loc[ticker,'Market_Cap']
        data_df.loc[ticker,'price']=last_price.loc[ticker]
    data_df.dropna(inplace=True)
    clean_tickers=data_df.index.to_list()
        
    return(clean_tickers, data_df)

def prune_tickers_df(tickers_old, tickers_new, tickers_df):
    for ticker in tickers_old:
        if ticker not in tickers_new:
            tickers_df.drop([ticker], axis=1, inplace=True)
    return(tickers_df)

def make_port_df(weights_df, clean_tickers, data_df, tickers_df, amount):
    port_df=weights_df.loc[weights_df[0]>0]
    port_df=port_df/port_df[0].sum()
    port_df.columns=['weights']
    port_df['price']=0
    port_df['returns']=0
    port_df['shares']=0
    port_df['value']=0

    
    return (fill_port_df(clean_tickers, port_df, data_df, tickers_df,amount))



    
def fill_port_df(clean_tickers, port_df, data_df, tickers_df,amount):

    port_tickers=port_df.index.to_list()
    tickers_df=prune_tickers_df(clean_tickers, port_tickers, tickers_df)
    last_price=tickers_df.iloc[-1,:]
    last_price.columns=['price']
    for ticker in port_tickers:
        port_df.loc[ticker,'price']=last_price.loc[ticker]
        port_df.loc[ticker,'returns']=data_df.loc[ticker,'returns']
        port_df.loc[ticker,'shares']=(amount*port_df.loc[ticker, 'weights']//port_df.loc[ticker, 'price'])
        port_df.loc[ticker,'value']=(port_df.loc[ticker, 'shares']*port_df.loc[ticker, 'price'])  
    left=amount-port_df['value'].sum()
    
    return(left, port_df, port_tickers)


def make_port_metrics(port_df, tickers_df):
    wts=port_df['weights'].to_list()
    wts=np.asarray(wts)
    port_ret=tickers_df.pct_change()
    port_ret.dropna(inplace=True)
    port_ret['return']=port_ret.mul(wts,axis=1).sum(axis=1)
    ann_ret=port_ret['return'].mean()* 252-.0062
    
    cov_matrix = port_ret.iloc[:,0:-1].cov()
    ann_port_std=round(((np.sqrt(np.dot(wts.T,np.dot(cov_matrix, wts)))* np.sqrt(252))),6)
    sharpe=ann_ret/ann_port_std
    
    metrics=[ann_ret, sharpe, ann_port_std]
    
    
    return (metrics)
    




'''This block of code is the Black Litterman portfolio functions, thy pyprotfolioopt, library I wanted to use would not compile on EC2.
this code was derived from https://github.com/overney/python/blob/master/Black%20Litterman%20Model.ipynb, and does not have the optimization 
that the other library achieved, this also discusses BL https://python-advanced.quantecon.org/black_litterman.html'''

def in_return_cov(df): 
    '''calculating the annual returns for each assset and generating a covariance matrix of the returns'''
    price = df.pct_change()
    price.iloc[0,:] = 0    
    
    returns = np.matrix(price)
    mean_returns = np.mean(returns, axis = 0)
    
    annual_returns = np.array([])
    for i in range(len(np.transpose(mean_returns))):
        annual_returns = np.append(annual_returns,(mean_returns[0,i]+1)**252-1)   
    
    cov = price.cov()*252
    
    return (annual_returns, np.matrix(cov))


def port_return(W,r):
    return sum(W*r)


def mkt_weights(weights):
    return np.array(weights) / sum(weights)

def risk_return(W,S,r,rf=0.0063):
    var = np.dot(np.dot(W,S),np.transpose(W))
    port_r = port_return(W,r)
    return np.ndarray.item((port_r - rf) / var)

def vector_equilibrium_return(S, W, r ):
    
    A = risk_return(W,S,r)
    
    return (np.dot(A,np.dot(S,W)))

def diago_omega(t, P, S):
    
    omega = np.dot(t,np.dot(P,np.dot(S,np.transpose(P))))
    
    for i in range(len(omega)):
        for y in range(len(omega)):
            if i != y: omega[i,y] = 0
    return omega

def make_view_matrix(clean_tickers, data_df):
    N=len(clean_tickers)
    Q = np.zeros((N,1))
    P = np.zeros((N,N))
    for n in range(len(clean_tickers)):
        P[n,n]=1
        Q[n,0]=data_df.iloc[n,0]
    return(P, Q)
    
def posterior_estimate_return(t,S,P,Q,PI):
    
    omega = diago_omega(t, P, S)
    
    parte_1 = t*np.dot(S,np.transpose(P))
    parte_2 = np.linalg.inv(np.dot(P*t,np.dot(S,np.transpose(P))) + omega)
    parte_3 = Q - np.dot(P,np.transpose(PI))
    
    return np.transpose(PI) + np.dot(parte_1,np.dot(parte_2,parte_3))

def posterior_covariance(t,S,P,PI):
    
    omega = diago_omega(t, P, S)
    
    parte_1 = t*np.dot(S,np.transpose(P))
    parte_2 = np.linalg.inv(t*np.dot(P,np.dot(S,np.transpose(P)))+omega)
    parte_3 = t*np.dot(P,S)

    return t*S - np.dot(parte_1,np.dot(parte_2,parte_3))




'''Here we package the lambda functions response and send it back to the client'''

def make_output (port_df, left, amount, metrics, port_tickers):
    '''Here we are making the text output for the chatbot, it will give all the metrics for the optimized portfolio'''
    port_str=''

    for ticker in port_tickers:
        str=(f' {int(port_df.loc[ticker, "shares"])} shares of {ticker} worth ${port_df.loc[ticker,"value"]:.2f},\n')
        port_str=port_str+str

    out_str=(f'''For your ${amount} investment, we have calculated the most afficient portfolio for your level of risk will be
    {port_str[:-3]} 
    and you will have ${left:.2f} leftover.
    This portfolio has a current annualized return of {metrics[0]*100:.2f}%, voluntility of {metrics[2]:.2f}, and a sharp raio of {metrics[1]:.2f}''')
    
    return(out_str)



def close(session_attributes, fulfillment_state, message):
    '''this packages up the full JSON response back to the chatbot'''
    response = {
        'sessionAttributes': session_attributes,
        'dialogAction': {
            'type': 'Close',
            'fulfillmentState': fulfillment_state,
            "message": {
                "contentType": "PlainText",
                "content": message 
                
            },
        }
    }
    return response





def get_slots(intent_request):
    """
    Fetch all the slots and their values from the current intent.
    """
    return intent_request["currentIntent"]["slots"]



### Intents Handlers ###
def make_portfolio(intent_request):
    """
    Performs fulfillment for recommending a portfolio.
    """

    '''Unpack the inputs from the chatbot'''
    bday = get_slots(intent_request)["bday"]
    amount = parse_float(get_slots(intent_request)["amount"])
    retire =parse_float( get_slots(intent_request)["retire"])
    risk = get_slots(intent_request)["risk"]
    income =parse_float(get_slots(intent_request)["income"])
    
    
    '''we will assess the clients risk adversion pull the csv for the crypto dataframe,
    create their portfolio, and optimize it'''
    client_risk=risk_assay(bday, income, amount, risk, retire)
    coin_df=get_coins()
    symbols=get_tickers(client_risk, coin_df)
    coin_df.set_index('symbol', inplace=True)
    tickers_df=get_portfolio(symbols)
    tickers=tickers_df.columns.to_list()

    
    
    '''Here we start the Black Litterman model, It needs better optimization'''
    asset_returns , S= in_return_cov(tickers_df)
    clean_tickers, data_df, tickers_df=make_data_df(coin_df, tickers_df, asset_returns, tickers)
    asset_returns , S= in_return_cov(tickers_df)
    W = mkt_weights(data_df['mrkcap'])
    A = risk_return(W,S,data_df['returns'])
    PI = vector_equilibrium_return(S, W, data_df['returns'])
    P, Q=make_view_matrix(clean_tickers, data_df)
    t = 1 / len(tickers_df)
    expc_return = posterior_estimate_return(t,S,P,Q,PI)
    post_cov = posterior_covariance(t,S,P,PI)
    cov_post_estimate = post_cov + S
    new_weight = np.dot(np.transpose(expc_return),np.linalg.inv(A*cov_post_estimate))
    weights_df=pd.DataFrame(new_weight, columns = clean_tickers).T
    
    
    
    '''using the weights we will allocate their investment'''
    left, port_df, port_tickers=make_port_df(weights_df, clean_tickers, data_df, tickers_df, amount)
    metrics=make_port_metrics(port_df, tickers_df)
    output_message=make_output(port_df, left, amount, metrics, port_tickers)
    
    
    return close(intent_request["sessionAttributes"], "Fulfilled", output_message)




### Intents Dispatcher ###
def dispatch(intent_request):
    """
    Called when the user specifies an intent for this bot.
    """

    # Get the name of the current intent
    intent_name = intent_request["currentIntent"]["name"]

    # Dispatch to bot's intent handlers
    if intent_name == "coinBot_test":
        return make_portfolio(intent_request)

    raise Exception("Intent with name " + intent_name + " not supported")


### Main Handler ###
def lambda_handler(event, context):
    """
    Route the incoming request based on intent.
    The JSON body of the request is provided in the event slot.
    """

    return dispatch(event)
