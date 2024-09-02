from gemini_agents_toolkit import agent
from gemini_agents_toolkit.pipeline.basic_step import BasicStep
from gemini_agents_toolkit.pipeline.if_step import IfStep
from gemini_agents_toolkit.pipeline.terminal_step import TerminalStep
from gemini_agents_toolkit.pipeline.summary_step import SummaryStep

import vertexai
import random


current_limit_buy_price = None
current_limit_sell_price = None


def check_if_limit_sell_order_exists():
    """Check if limit sell order exists"""
    global current_limit_sell_price
    return current_limit_sell_price is not None


def cancel_limit_sell_order():
    """Cancel limit sell order"""
    global current_limit_sell_price
    current_limit_sell_price = None
    return "Limit sell order canceled"


def set_limit_sell_order(price):
    """Set limit sell order
    
    Args:
        price: float
            Price to set limit sell order"""
    global current_limit_sell_price
    current_limit_sell_price = price
    return f"Limit sell order set at {price}"


def check_current_tqqq_price():
    """Current price of TQQQ"""
    # return 125.3 + random number in the range [-100, 100]
    return 125.3 + random.randint(-100, 100)
    

def get_current_limit_buy_price():
    """Current limit buy price of TQQQ"""
    global current_limit_buy_price
    return current_limit_buy_price


def check_if_limit_buy_order_exists():
    """Check if limit buy order exists"""
    global current_limit_buy_price
    return current_limit_buy_price is not None


def cancel_limit_buy_order():
    """Cancel limit buy order"""
    global current_limit_buy_price
    current_limit_buy_price = None
    return "Limit buy order canceled"


def set_limit_buy_order(price):
    """Set limit buy order
    
    Args:
        price: float
            Price to set limit buy order"""
    global current_limit_buy_price
    current_limit_buy_price = price
    return f"Limit buy order set at {price}"


def check_how_many_shares_i_own():
    """Check how many shares of TQQQ I own"""
    return 30 + random.randint(-20, 20)


vertexai.init(project="gemini-trading-backend", location="us-west1")

all_functions = [check_if_limit_sell_order_exists, cancel_limit_sell_order, set_limit_sell_order, check_current_tqqq_price, check_if_limit_buy_order_exists, get_current_limit_buy_price, cancel_limit_buy_order, set_limit_buy_order, check_how_many_shares_i_own]
investor_agent = agent.create_agent_from_functions_list(functions=all_functions, model_name="gemini-1.5-pro")

check_price_and_set_sell_limit = (BasicStep(investor_agent, "check current price of TQQQ")
                                  .next_step("set limit sell order for TQQQ for price +4% of current price")
                                  .final_step_summary())
check_if_limit_sell_existst_step = IfStep(investor_agent, "is there a limit sell exists already", if_step=SummaryStep(investor_agent), else_step=check_price_and_set_sell_limit)

set_limit_buy_order_step = (BasicStep(investor_agent, "check current price of TQQQ")
                            .next_step("set limit buy order for TQQQ for price 3 precent below the current price. Do not return compute formula, do compute of the price yourself in your head")
                            .final_step_summary())

recreate_limit_buy_order_step = BasicStep(investor_agent, "cancel limit buy order")
recreate_limit_buy_order_step.next_step("set limit buy order for TQQQ for price 3 precent below the current price. Do not return compute formula, do compute of the price yourself in your head")

maybe_recreate_limit_buy_order_step = IfStep(investor_agent, "is there current limit buy price lower than curent price of TQQQ -5%?", if_step=recreate_limit_buy_order_step, else_step=SummaryStep(investor_agent))

check_if_limit_buy_order_exists_step = IfStep(investor_agent, "is there a limit buy exists already", if_step=maybe_recreate_limit_buy_order_step, else_step=set_limit_buy_order_step)
check_if_i_own_30_shares_step = IfStep(investor_agent, "chec if I own more than 30 shares of TQQQ", if_step=check_if_limit_sell_existst_step, else_step=check_if_limit_buy_order_exists_step)

print(check_if_i_own_30_shares_step.execute())
