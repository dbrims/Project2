import json

### Required Libraries ###
from datetime import datetime
from dateutil.relativedelta import relativedelta
from botocore.vendored import requests


### Functionality Helper Functions ###

def parse_float(n):
    """
    Securely converts a non-numeric value to float.
    """
    try:
        return float(n)
    except:
        return "error"
    
    
def check_bday(bday):
    birth_date = datetime.strptime(bday, "%Y-%m-%d")
    age = relativedelta(datetime.now(), birth_date).years
    if age<21:
        return False
    else:
        return True


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


def validate_data(bday, amount, risk, retire, income):
    """
    Validates the data provided by the user.
    """

    # Validate that the user is over 21 years old
    if bday is not None:
        birth_date = datetime.strptime(bday, "%Y-%m-%d")
        age = relativedelta(datetime.now(), birth_date).years
        if age > 110 or age<=0:
            return build_validation_result(
                False,
                "bday",
                f"While you look good for age {age}, "
                "please enter your real birthdate.",
            )

    # Validate the investment amount, it should be > 0
    if income is not None:
        income = parse_float(income)  # Since parameters are strings it's important to cast values
        if income <= 0:
            return build_validation_result(
                False,
                "income",
                "I think you mistyped your income, "
                "please reenter a positive number.",
            )
        
    if amount is not None:
        amount = parse_float(amount)  # Since parameters are strings it's important to cast values
        if amount <= 0:
            return build_validation_result(
                False,
                "amount",
                "I think you mistyped the amount you want to invest, "
                "please reenter a positive number.",
            )
        
    if risk is not None:
        if risk.lower()=="low" or risk.lower()=="medium" or risk.lower()=="high":
            pass
        else:
            return build_validation_result(
                False,
                "risk",
                "I think you mistyped your level of risk aversion, "
                "please enter low, medium or high.",
            )
        
    if retire is not None:
        retire = parse_float(retire)  # Since parameters are strings it's important to cast values
        if retire <= 0:
            return build_validation_result(
                False,
                "retire",
                "I did not quite get that, unless you retired before you were even born,"
                "please enter a positive number.",
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


def close(session_attributes, fulfillment_state):
    """
    Defines a close slot type response.
    """

    response = {
        "sessionAttributes": session_attributes,
        "dialogAction": {
            "type": "Close",
            "fulfillmentState": "Failed",
            "message": {
                "contentType": "PlainText",
                "content": "We told you that you must be at least 21 to use this service. "
                           "Please come back when you are old enough."},
        },
    }

    return response


### Intents Handlers ###
def validating(intent_request):
    """
    Performs dialog management and fulfillment for recommending a portfolio.
    """

    # Gets slots' values
    bday = get_slots(intent_request)["bday"]
    amount =get_slots(intent_request)["amount"]
    retire =get_slots(intent_request)["retire"]
    risk = get_slots(intent_request)["risk"]
    income =get_slots(intent_request)["income"]

    # Gets the invocation source, for Lex dialogs "DialogCodeHook" is expected.
    source = intent_request["invocationSource"]  #

    if source == "DialogCodeHook":
        # This code performs basic validation on the supplied input slots.

        # Gets all the slots
        slots = get_slots(intent_request)

        # Validates user's input using the validate_data function
        validation_result = validate_data(bday, amount, risk, retire, income)

        # If the data provided by the user is not valid,
        # the elicitSlot dialog action is used to re-prompt for the first violation detected.
        if not validation_result["isValid"]:
            if not check_bday(bday):
                return close (intent_request["sessionAttributes"], "fullfilled")
            else:
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


# ### Intents Dispatcher ###
def dispatch(intent_request):
    """
    Called when the user specifies an intent for this bot.
    """

    # Get the name of the current intent
    intent_name = intent_request["currentIntent"]["name"]

    # Dispatch to bot's intent handlers
    if intent_name == "coinBot":
        return validating(intent_request)

    raise Exception("Intent with name " + intent_name + " not supported")




### Main Handler ###
def lambda_handler(event, context):
    """
    Route the incoming request based on intent.
    The JSON body of the request is provided in the event slot.
    """

    return dispatch(event)