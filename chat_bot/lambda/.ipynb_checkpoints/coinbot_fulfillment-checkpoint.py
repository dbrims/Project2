### Required Libraries ###
from datetime import datetime
from dateutil.relativedelta import relativedelta
# from botocore.vendored import requests
# !pip install PyPortfolioOpt
# !pip install pulp
# !pip install pulp
# !pip install ccxt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
# import boto3

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
    bucket = 'OurS3'
    key = 'OurFile.csv'
    response = s3.get_object(Bucket=bucket, Key=key)
    df=pd.read_csv(response)
    
    return(df)



def get_tickers(ave_risk, df):
    '''we are saying cryto class have a risk associated with them and based 
    on the risk metric for the client we are assigning them one of those risk 
    portfolios'''
    tickers=[]
    if ave_risk==1:
        tickers=df['ticker'].loc[df['class']==x].to_list()
    elif ave_risk==2:
        tickers=df['ticker'].loc[df['class']==x].to_list()
    elif ave_risk==3:
        tickers=df['ticker'].loc[df['class']==x].to_list()
        
    return(tickers)



def get_portfolio(tickers):
    i=0
    for ticker in tickers:
        if i==0:
            data = exchange.fetch_ohlcv(ticker, '1d')
            header = ['Date', 'Open', 'High', 'Low', ticker, 'Volume']
            ticker_df = pd.DataFrame(data, columns=header)
            ticker_df['Date']=pd.to_datetime(ticker_df.Date/1000, unit='s')
            ticker_df.set_index('Date', inplace=True)
            ticker_df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)
            i+=1
        else:
            data = exchange.fetch_ohlcv(ticker, '1d')
            header = ['Date', 'Open', 'High', 'Low', ticker, 'Volume']
            df = pd.DataFrame(data, columns=header)
            df['Date']=pd.to_datetime(df.Date/1000, unit='s')
            df.set_index('Date', inplace=True)
            df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)
            ticker_df=pd.concat([ticker_df, df], axis=1, join='inner')
            
    return(ticker_df)
    
def eff_frontier(df):
    '''using the pyportfolioopt package in python, we will optimize the weights for the portforlio and give the current annual return, sharpe ratio and volatility'''
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights() 
    performance=ef.portfolio_performance(verbose=False)
    
    return(cleaned_weights, performance)



def allocation(df, cleaned_weights, amount):
    '''using the weights we created we will allocate the clients investment, giving them 
    how many coins to buy and what the value of those coins would be (as well as any 
    left over cash in their account)'''
    
    portfolio={}
    latest_prices = get_latest_prices(df)
    weights = cleaned_weights 
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=amount)
    allocation, leftover = da.lp_portfolio()
    tickers=allocation.keys()
    last_price=df.iloc[-1,:]
    for ticker in tickers:
        price=last_price.loc[ticker]
        portfolio[ticker]={}
        portfolio[ticker]['shares']=allocation[ticker]
        portfolio[ticker]['value']=allocation[ticker]*price
        
    return(portfolio, leftover)



def make_output (portfolio, leftover, amount, performance):
    '''Here we are making the text output for the chatbot, it will give all the metrics for the optimized portfolio'''
    port_str=''
    
    for key in portfolio.keys():
        str=(f' {int(portfolio[key]["shares"])} shares of {key} worth ${portfolio[key]["value"]:.2f},')
        port_str=port_str+str
        
    out_str=(f'''For your {amount} investment, we have calculated the most afficient portfolio for your level of risk
    will be {port_str[:-1]} and you will have ${leftover:.2f} leftover.  This portfolio has a current annualized return of
    {performance[0]*100:.2}%, voluntility of {performance[1]:.2}, and a sharp raio of {performance[2]:.2}''')
    
    return(out_str)



def close(bday, session_attributes, fulfillment_state, message):
    '''this packages up the full JSON response back to the chatbot'''
        response = {
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Close',
                'fulfillmentState': fulfillment_state,
                'message': message
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

    # Gets slots' values
    bday = get_slots(intent_request)["bday"]
    amount = parse_float(get_slots(intent_request)["amount"])
    retire =parse_float( get_slots(intent_request)["retire"])
    risk = get_slots(intent_request)["risk"]
    income =parse_float(get_slots(intent_request)["income"])
    
    '''we will assess the clients risk adversion pull the csv for the crypto dataframe,
    create their portfolio, and optimize it'''
    client_risk=risk_assay(bday, income, amount, risk, retire)
    coin_df=get_coins()
    tickers=get_tickers(client_risk, coin_df)
    port_df=get_portfolio(tickers)
    weights, performance=eff_frontier(port_df)
    portfolio, leftover=allocation(port_df, weights, amount)
    output_message=make_output(portfolio, leftover, amount, performance)
    
    
    return close(bday, intent_request["sessionAttributes"], "Fulfilled", output_message)


### Intents Dispatcher ###
def dispatch(intent_request):
    """
    Called when the user specifies an intent for this bot.
    """

    # Get the name of the current intent
    intent_name = intent_request["currentIntent"]["name"]

    # Dispatch to bot's intent handlers
    if intent_name == "coinBot":
        return make_portforlio(intent_request)

    raise Exception("Intent with name " + intent_name + " not supported")


### Main Handler ###
def lambda_handler(event, context):
    """
    Route the incoming request based on intent.
    The JSON body of the request is provided in the event slot.
    """

    return dispatch(event)
