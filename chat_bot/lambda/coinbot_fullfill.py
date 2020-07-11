### Required Libraries ###
from datetime import datetime
from dateutil.relativedelta import relativedelta
from botocore.vendored import requests
!pip install PyPortfolioOpt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

### Functionality Helper Functions ###
def parse_float(n):
    """
    Securely converts a non-numeric value to float.
    """
    try:
        return float(n)
    except ValueError:
        return float("nan")
    
def risk assay(age, income, amount, risk, retire):
    '''from the clients inputs we are going to assign them a risk score high risk=2, moderate risk=2, low risk=1'''

    rlst=[]
    working_age=retire-age
    income_ratio=income/amount
    
    '''we are going to use their age and retirement age to create a risk value, closer to retirement, the higher the risk'''
    if working_age<17:
        rlist.append(3)
    elif working_age>=17 and working_age<34:
        rlst.append(2)
    else:
        rlst.append(1)
        
    '''we will use their income and amount to invest.  If the investment amount is a sizable chunk of their annual income
    we will say it is higher risk'''
    if income_ratio<2:
        rlist.append(3)
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
    else:
    """ not sure how to do an input validation... need to work on it"""
        
    ave_risk = round(mean(rlst),0)
    return(ave_risk)

def get_portfolio(ave_risk,df):
    '''we are saying cryto class have a risk associated with them and based 
    on the risk metric for the client we are assigning them one of those risk 
    portfolios'''
    tickers=[]
    if ave_risk==1:
        tickers=df['ticker'].loc[df['class']==x].to_list()
    elif ave_risk==2:
        tickers=df['ticker'].loc[df['class']==x].to_list()
    elif ave_risk==:
        tickers=df['ticker'].loc[df['class']==x].to_list()
        
    return(tickers)

def get_portfolio(tickers):
    '''we are taking our list of tickers and generating a dataframe of the assets'''
    i=0
    for ticker in tickers:
        if i==0:
            ticker_df= ''' get data from API'''
            ticker_df.set_index('column name', inplace=True)
            ticker_df=ticker_df.drop(columns=['x','y'])
            ticker_df.columns=[ticker]
            i+=1
        else:
            df= ''' get data from API'''
            df.set_index('date column name', inplace=True)
            df=ticker_df.drop(columns=['x','y'])
            df.columns=[ticker]
            ticker_df=pd.concat([ticker_df, df], axis=1, join='inner')
        return(ticker_df)
    
def eff_frontier(ticker_df):
    '''using the pyportfolioopt package in python, we will optimize the weights for the portforlio and give the current annual return, sharpe ratio and volatility'''
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights() 
    performance=ef.portfolio_performance(verbose=True)
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
    for col in range(len(tickers)):
        price=df.iloc[-1,col]
        portfolio[ticker[col]]['shares']=allocation[tickers[col]]
        portfolio[ticker[col]]['value']=allocation[tickers[col]]*price
        
    return(portfolio, leftover)

def make_output (portfolio, leftover, amount, performance):
    port_str=''
    for key in portfolio:
        str=f' {portfolio[key]["shares"]} shares of {portfolio[key]["shares"]} worth {portfolio[key]["value"]},'
        port_str=port_str+str
        
    out_str=f'For your {amount} investment, we have calculated the most afficient portfolio for your level of risk\n
    will be {port_str[:-1]} and you will have ${leftover} leftover.  This portfolio has a current annualized return of\n
    {performance[0]*100:.2}%, voluntility of {performance[1]:.2}, and a sharp raio of {performance[2]:.2}'
    return(out_str)
    
        
    
            

def get_btcprice():
    """
    Retrieves the current price of bitcoin in US Dollars from the alternative.me Crypto API.
    """
    bitcoin_api_url = "https://api.alternative.me/v2/ticker/bitcoin/?convert=USD"
    response = requests.get(bitcoin_api_url)
    response_json = response.json()
    price_usd = parse_float(response_json["data"]["1"]["quotes"]["USD"]["price"])
    return price_usd


def build_validation_result(is_valid, violated_slot, message_content):
    """
    Defines an internal validation message structured as a python dictionary.
    """
    if message_content is None:
        return {"isValid": is_valid, "violatedSlot": violated_slot}

    return {
        "isValid": is_valid,
        "violatedSlot": violated_slot,
        "message": {"contentType": "PlainText", "content": message_content},
    }


def validate_data(birthday, usd_amount, intent_request):
    """
    Validates the data provided by the user.
    """

    # Validate that the user is over 21 years old
    if birthday is not None:
        birth_date = datetime.strptime(birthday, "%Y-%m-%d")
        age = relativedelta(datetime.now(), birth_date).years
        if age < 21:
            return build_validation_result(
                False,
                "birthday",
                "You need to be at least 21 years old to use this service, "
                "please come back when you are old enough.",
            )

    # Validate the investment amount, it should be > 0
    if usd_amount is not None:
        usd_amount = parse_float(
            usd_amount
        )  # Since parameters are strings it's important to cast values
        if usd_amount <= 0:
            return build_validation_result(
                False,
                "usdAmount",
                "The amount to convert should be greater than zero, "
                "please provide a correct amount in USD to convert.",
            )

    # A True results is returned if age or amount are valid
    return build_validation_result(True, None, None)


### Dialog Actions Helper Functions ###
def get_slots(intent_request):
    """
    Fetch all the slots and their values from the current intent.
    """
    return intent_request["currentIntent"]["slots"]


def elicit_slot(session_attributes, intent_name, slots, slot_to_elicit, message):
    """
    Defines an elicit slot type response.
    """

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {
            "type": "ElicitSlot",
            "intentName": intent_name,
            "slots": slots,
            "slotToElicit": slot_to_elicit,
            "message": message,
        },
    }


def delegate(session_attributes, slots):
    """
    Defines a delegate slot type response.
    """

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {"type": "Delegate", "slots": slots},
    }


def close(session_attributes, fulfillment_state, message):
    """
    Defines a close slot type response.
    """

    response = {
        "sessionAttributes": session_attributes,
        "dialogAction": {
            "type": "Close",
            "fulfillmentState": fulfillment_state,
            "message": message,
        },
    }

    return response


### Intents Handlers ###
def convert_usd(intent_request):
    """
    Performs dialog management and fulfillment for recommending a portfolio.
    """

    # Gets slots' values
    birthday = get_slots(intent_request)["birthday"]
    usd_amount = get_slots(intent_request)["usdAmount"]

    # Gets the invocation source, for Lex dialogs "DialogCodeHook" is expected.
    source = intent_request["invocationSource"]  #

    if source == "DialogCodeHook":
        # This code performs basic validation on the supplied input slots.

        # Gets all the slots
        slots = get_slots(intent_request)

        # Validates user's input using the validate_data function
        validation_result = validate_data(birthday, usd_amount, intent_request)

        # If the data provided by the user is not valid,
        # the elicitSlot dialog action is used to re-prompt for the first violation detected.
        if not validation_result["isValid"]:
            slots[validation_result["violatedSlot"]] = None  # Cleans invalid slot

            # Returns an elicitSlot dialog to request new data for the invalid slot
            return elicit_slot(
                intent_request["sessionAttributes"],
                intent_request["currentIntent"]["name"],
                slots,
                validation_result["violatedSlot"],
                validation_result["message"],
            )

        # Fetch current session attributes
        output_session_attributes = intent_request["sessionAttributes"]

        # Once all slots are valid, a delegate dialog is returned to Lex to choose the next course of action.
        return delegate(output_session_attributes, get_slots(intent_request))

    # Get the current price of BTC in USD and make the conversion from USD to BTC.
    btc_value = parse_float(usd_amount) / get_btcprice()
    btc_value = round(btc_value, 2)

    # Return a message with conversion's result.
    return close(
        intent_request["sessionAttributes"],
        "Fulfilled",
        {
            "contentType": "PlainText",
            "content": """Thank you for your information;
            you can get {} Bitcoins for your {} US Dollars.
            """.format(
                btc_value, usd_amount
            ),
        },
    )


### Intents Dispatcher ###
def dispatch(intent_request):
    """
    Called when the user specifies an intent for this bot.
    """

    # Get the name of the current intent
    intent_name = intent_request["currentIntent"]["name"]

    # Dispatch to bot's intent handlers
    if intent_name == "ConvertUSD":
        return convert_usd(intent_request)

    raise Exception("Intent with name " + intent_name + " not supported")


### Main Handler ###
def lambda_handler(event, context):
    """
    Route the incoming request based on intent.
    The JSON body of the request is provided in the event slot.
    """

    return dispatch(event)
