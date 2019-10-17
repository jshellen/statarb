# -*- coding: utf-8 -*-

def signed_quantity(action,quantity):
    
    if(not isinstance(quantity,int)):
        raise ValueError('Input quantity has to be an integer!')  
    if(not isinstance(action,str)):
        raise ValueError('Input action has to be a string!')
    out = None
    if(action=="SLD"):
        out = -quantity
    elif(action=="BOT"):
        out =  quantity
    else:
        raise ValueError("Action has to be either BOT or SLD")
    return out

def action_and_quantity(x):
    
    if(not isinstance(x,int)):
        raise ValueError('Input has to be an integer')
    if(x<0):
        return 'SLD',abs(x)
    elif(x>0):
        return 'BOT',abs(x)
    else:
        return None,None
    
def infer_trade_action(action):
    
    out = None
    if(isinstance(action,str)):
        if(action == "BOT"):
            out = 1
        elif(action == "SLD"):
            out = 2
        else:
            raise ValueError(f'{action} is not valid action!')
    else:
        if(action not in [1,2]):
            raise ValueError(f'{action} is not valid action!')
        else:
            out = action
    return out

