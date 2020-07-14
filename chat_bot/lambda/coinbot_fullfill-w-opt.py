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
from pathlib import Path


public_key=os.environ.get(KRAKEN_API_KEY)
secret_key=os.environ.get(KRAKEN_SECRET_API_KEY)


s3 = boto3.client('s3')


###############################


'''data is imported as strings, so have to convert numbers with this helper fuction'''
def parse_float(n):
    """
    Securely converts a non-numeric value to float.
    """
    try:
        return float(n)
    except ValueError:
        return "error"
    

###############################
    
'''In this section I am pulling in all the data we will need to fulfill the request'''

def get_slots(intent_request):
    """
    Fetch all the slots and their values from the current intent.
    """
    return intent_request["currentIntent"]["slots"]


def get_coins():
    
    '''here we read the coins csv file from the E3 drive and pass it back
    you will have to update the bucket name with what you named your 
    bucket'''
    bucket = 'ft-project-2'
    key = 'Classification.csv'
    response = s3.get_object(Bucket=bucket, Key=key)
    csv_file=response['Body']
    df=pd.read_csv(csv_file)
#     data=Path('Classification.csv')
#     cdf=pd.read_csv(data)

    return(cdf)


'''we are using the free kraken api to get our crypto data'''
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



############################

''' in this section we start to aggregate the data and get it in useable form to throw into the 
portfolio optimization model'''


'''here I am taking the client inputs and assigning them an level of risk.  For this situation, the assignment is 
basically arbitary as to what cutoffs are used and how they all play togehter'''
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

'''I am compiling all the single point metrics for each cyrpto in a single dataframe, which 
will get updated as the lambda function does its thing'''
    
def make_data_df(coin_df, tickers_df, asset_returns, symbols):
    tickers=tickers_df.columns.tolist()
    data_df=pd.DataFrame()
    data_df['tickers']=tickers
    data_df['returns']=asset_returns
    data_df.set_index('tickers', inplace=True)
    data_df['ann_ret']=0
    data_df['SD']=0
    data_df['mrkcap']=0
    data_df['price']=0

    clean_tickers, data_df=fill_data_df(coin_df, data_df, tickers_df, tickers)
    tickers_df=prune_tickers_df(tickers,clean_tickers, tickers_df)
    return(clean_tickers, data_df, tickers_df)

def fill_data_df(coin_df,data_df, tickers_df, tickers):
    last_price=tickers_df.iloc[-1,:]
    rets_df=tickers_df.pct_change()
    rets=rets_df.mean() * 252
    SD=rets_df.std() * np.sqrt(252)
    
    for ticker in tickers:
        data_df.loc[ticker,'SD']=SD.loc[ticker]
        data_df.loc[ticker,'ann_ret']=rets.loc[ticker]        
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


def update_data_df(expc_return, min_vol, clean_tickers, data_df, amount):
    data_df['opt_wt']=0
    data_df['exp_return']=0
    data_df['shares']=0
    data_df['value']=0
    
    return (data_df_add(expc_return, min_vol, clean_tickers, data_df, amount))


    
def data_df_add(expc_return, min_vol, clean_tickers, data_df, amount):
    data_df['exp_return']=expc_return
    data_df['opt_wt']=min_vol['x']
    data_df['port_ret']=data_df['exp_return']*data_df['opt_wt']
    for ticker in clean_tickers:
        data_df.loc[ticker,'shares']=(amount*data_df.loc[ticker, 'opt_wt']//data_df.loc[ticker, 'price'])
        data_df.loc[ticker,'value']=(data_df.loc[ticker, 'shares']*data_df.loc[ticker, 'price'])  
    left=amount-data_df['value'].sum()
    
    return(left, data_df)
    

####################


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

'''I have two different methods to calcuate the omega, in the original they are using the cov
for the portfolio as the uncertainty of the views, while in the second I am using the 
standard deviation of the indiviual crypto as a measure of the uncertainty of that crysto'''

def diago_omega(t, P, S):
    
    omega = np.dot(t,np.dot(P,np.dot(S,np.transpose(P))))
    
    for i in range(len(omega)):
        for y in range(len(omega)):
            if i != y: omega[i,y] = 0
    return omega


def my_omega(data_df, clean_tickers):
    intervals=[]
    variances=[]
    for ticker in clean_tickers:
        interval=(
            data_df.loc[ticker,'ann_ret']
            -data_df.loc[ticker,'SD'], 
            data_df.loc[ticker, 'ann_ret']
            +data_df.loc[ticker, 'SD']
    )
        intervals.append(interval)
    for lb, ub in intervals:
        sigma = (ub - lb)/2
        variances.append(sigma ** 2)
    omega=np.diag(variances)
    return(omega)

'''there are several ways to make views, I am just doing the easiest
and using the annualized average return as an absolute view
rather then trying to do relative views.... easier to do as HTP'''
def make_view_matrix(clean_tickers, data_df):
    N=len(clean_tickers)
    Q = np.zeros((N,1))
    P = np.zeros((N,N))
    for n in range(len(clean_tickers)):
        P[n,n]=1
        Q[n,0]=data_df.iloc[n,0]
    return(P, Q)
    
def posterior_estimate_return(t,S,P,Q,PI,omega):
    
#     omega = diago_omega(t, P, S)
    
    parte_1 = t*np.dot(S,np.transpose(P))
    parte_2 = np.linalg.inv(np.dot(P*t,np.dot(S,np.transpose(P))) + omega)
    parte_3 = Q - np.dot(P,np.transpose(PI))
    
    return np.transpose(PI) + np.dot(parte_1,np.dot(parte_2,parte_3))

def posterior_covariance(t,S,P,PI,omega):
    
#     omega = diago_omega(t, P, S)
    
    parte_1 = t*np.dot(S,np.transpose(P))
    parte_2 = np.linalg.inv(t*np.dot(P,np.dot(S,np.transpose(P)))+omega)
    parte_3 = t*np.dot(P,S)

    return t*S - np.dot(parte_1,np.dot(parte_2,parte_3))


####################
'''in this section I am optimizing the results of the black litterman porfolio to create the final weights
I am also generating the metrics used to feed back to the client'''

'''optimizing the weights of the protfolio by mean variance optimization. following this 
methodology https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f'''
def portfolio_annualised_performance(W, expc_return, cov_post_estimate):
    returns = np.sum(expc_return*W )
    std = np.sqrt(np.dot(W, np.dot(cov_post_estimate, W).T))
    return std, returns

def portfolio_volatility(W, expc_return, cov_post_estimate):
    return portfolio_annualised_performance(W, expc_return, cov_post_estimate)[0]

def min_variance(expc_return, cov_post_estimate):
    num_assets = len(expc_return)
    args = (expc_return, cov_post_estimate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def make_port_metrics(data_df, cov_post_estimate):
    ret=data_df['port_ret'].sum()
    std=(np.sqrt(np.dot(data_df['opt_wt'],np.dot(cov_post_estimate, data_df['opt_wt']).T))).tolist()
    sharpe=ret/std
    p_std=round(std[0][0],6)
    p_sharpe=round(sharpe[0][0],6)
    p_ret=round(ret, 6)
    metrics=[p_ret, p_sharpe, p_std]
    
    return (metrics)

#########


'''In this section we are packaging the output of the model and getting it ready to send 
to the client, to fulfill the request'''


'''Here we package the lambda functions response and send it back to the client'''

def make_output (port_df, left, amount, metrics, port_tickers):
    '''Here we are making the text output for the chatbot, it will give all the metrics for the optimized portfolio'''
    port_str=''
    
    if len(port_tickers)>0:
        for ticker in port_tickers:
            str=(f' {int(port_df.loc[ticker, "shares"])} shares of {ticker} worth ${port_df.loc[ticker,"value"]:.2f},\n')
            port_str=port_str+str

        out_str=(f'''For your ${amount} investment, we have calculated the most afficient portfolio for your level of risk will be
        {port_str[:-3]} 
        and you will have ${left:.2f} leftover.
        This portfolio has a current annualized return of {metrics[0]*100:.2f}%, voluntility of {metrics[2]:.2f}, and a sharp raio of {metrics[1]:.2f}
        Thank you for using the HAL3000 coinBot, Have a good day, Dave!''')
        
    else:
        out_str=(f'''Due to market conditions, we are unable to build a suitable portfolio of cyptos for you.  We suggest, at this time, you either 
        hold onto your ${amount} or invest in other asset classs.  
        Thank you for using the HAL3000 coinBot, Have a good day, Dave!''')
    
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

#########
'''here is the main body of the lambda function which is calling all the 
funciton above'''


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
    asset_returns , S= in_return_cov(tickers_df)  #realigning the dim of S and the portfolio
    W = mkt_weights(data_df['mrkcap'])
    A = risk_return(W,S,data_df['returns'])
    PI = vector_equilibrium_return(S, W, data_df['returns'])
    P, Q=make_view_matrix(clean_tickers, data_df)
    t = 1 / len(tickers_df)
    omega=my_omega(data_df, clean_tickers)
    expc_return = posterior_estimate_return(t,S,P,Q,PI,omega)
    post_cov = posterior_covariance(t,S,P,PI,omega)
    cov_post_estimate = post_cov + S
    
    
    
    '''Here we do the portfolio optimization and package the metrics'''
    min_vol = min_variance(expc_return, cov_post_estimate)
    left, data_df =update_data_df(expc_return, min_vol, clean_tickers, data_df, amount)
    metrics=make_port_metrics(data_df, cov_post_estimate)
    port_df=data_df.loc[data_df['shares']>0]
    port_tickers=port_df.index.to_list()

    '''Generating the output and shipping it back to the client, FULFILLED'''
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
