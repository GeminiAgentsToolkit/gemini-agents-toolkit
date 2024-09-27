"""This example illustrates creating an advanced pipeline with different step types."""

import random
import vertexai
from config import (PROJECT_ID, REGION, DEFAULT_MODEL, SIMPLE_MODEL)
from gemini_agents_toolkit import agent
from gemini_agents_toolkit.pipeline.eager_pipeline import EagerPipeline
from gemini_agents_toolkit.history_utils import summarize

# pylint: disable-next=invalid-name
current_limit_buy_price = None
# pylint: disable-next=invalid-name
current_limit_sell_price = None


def check_if_limit_sell_order_exists():
    """Check if limit sell order exists"""
    return current_limit_sell_price is not None


def cancel_limit_sell_order():
    """Cancel limit sell order"""
    # pylint: disable-next=global-statement
    global current_limit_sell_price
    current_limit_sell_price = None
    return "Limit sell order canceled"


def set_limit_sell_order(price: float):
    """Set limit sell order.
    Price is float, and it can not be a formula or logic that prints formula,
    input HAS to be fully computed price.
    
    Args:
        price: Price to set limit sell order"""
    # pylint: disable-next=global-statement
    global current_limit_sell_price
    current_limit_sell_price = price
    return f"Limit sell order set at {price}"


def check_current_tqqq_price():
    """Current price of TQQQ"""
    # return 125.3 + random number in the range [-100, 100]
    return 125.3 + random.randint(-100, 100)


def get_current_limit_buy_price():
    """Current limit buy price of TQQQ"""
    return current_limit_buy_price


def check_if_limit_buy_order_exists():
    """Check if limit buy order exists"""
    return current_limit_buy_price is not None


def cancel_limit_buy_order():
    """Cancel limit buy order"""
    # pylint: disable-next=global-statement
    global current_limit_buy_price
    current_limit_buy_price = None
    return "Limit buy order canceled"


def set_limit_buy_order(price: float):
    """Set limit buy order.
    Price is float, and it can not be a formula or logic that prints formula,
    input HAS to be fully computed price.
    
    Args:
        price: Price to set limit buy order"""
    # pylint: disable-next=global-statement
    global current_limit_buy_price
    current_limit_buy_price = price
    return f"Limit buy order set at {price}"


def check_how_many_shares_i_own():
    """Check how many shares of TQQQ I own"""
    return 30 + random.randint(-20, 1)


vertexai.init(project=PROJECT_ID, location=REGION)

all_functions = [
    check_if_limit_sell_order_exists,
    cancel_limit_sell_order,
    set_limit_sell_order,
    check_current_tqqq_price,
    check_if_limit_buy_order_exists,
    get_current_limit_buy_price,
    cancel_limit_buy_order,
    set_limit_buy_order,
    check_how_many_shares_i_own
]
investor_agent = agent.create_agent_from_functions_list(functions=all_functions,
                                                        model_name=DEFAULT_MODEL,
                                                        system_instruction="""You are an investor agent that can run set of actions.
                                                        when you are executing the actions, do not try to call multipele meathod at once.
                                                        You can only call one method at a time. But you can call a mathod, got a response and call another mathod.
                                                        You can NOT execute arbitary pythong code, you ONLY can call maethods/tools avialbe to you.""")

pipeline = EagerPipeline(default_agent=investor_agent, use_convert_to_bool_agent=True)
main_events_history = []
full_history = []
own_30_shares, _ = pipeline.boolean_step("do I own more than 30 shares of TQQQ")
if not own_30_shares:
    is_there_a_limit_sell_order, bool_result_history = pipeline.boolean_step("is there a limit sell order exists already?")
    full_history.extend(bool_result_history)
    _, history = pipeline.step("check current price of TQQQ")
    full_history.extend(history)
    main_events_history.extend(history)
    if not is_there_a_limit_sell_order:
        _, history = pipeline.step("set limit sell order for TQQQ for price +4% of current price", history=main_events_history)
        full_history.extend(history)
    else:
        is_there_a_limit_buy_order, history = pipeline.boolean_step("is there a limit buy order exists already?")
        full_history.extend(history)
        if is_there_a_limit_buy_order:
            if not pipeline.boolean_step(
                    "is there current limit buy price lower than current price of TQQQ -5%?"):
                _, history = pipeline.step("cancel limit buy order for TQQQ")
                full_history.extend(history)
                main_events_history.extend(history)
                _, history = pipeline.step(
                    """set limit buy order for TQQQ for price 3 percent below the current price. 
                    Do not return compute formula,
                    do compute of the price yourself in your head""", history=main_events_history)
                full_history.extend(history)
                main_events_history.extend(history)
        else:
            _, history = pipeline.step(
                """set limit buy order for TQQQ for price 3 percent below the current price. 
                Do not return compute formula, do compute of the price yourself in your head""")
            full_history.extend(history)
answer, _ = summarize(agent=investor_agent, history=full_history)
print(answer)
