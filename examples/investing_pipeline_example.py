"""This example illustrates creating an advanced pipeline with different step types."""

import random
from gemini_agents_toolkit.config import DEFAULT_MODEL
from gemini_agents_toolkit import agent
from gemini_agents_toolkit.pipeline import Pipeline
from google.adk.agents import LlmAgent

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
    return 30 + random.randint(-5, 5)


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
investor_agent = agent.ADKAgentService(
    agent=LlmAgent(model=DEFAULT_MODEL, name="investing_agent", tools=all_functions))

pipeline = Pipeline(default_agent=investor_agent, use_convert_agent_helper=True)
_, history_with_price = pipeline.step("check current price of TQQQ")
if pipeline.boolean_step("do I own more than 30 shares of TQQQ")[0]:
    pipeline.if_step("is there NO limit sell order exists already?",
                    then_steps=[
                        "set limit sell order for TQQQ for price +4% of current price",
                    ],
                    events=history_with_price)
else:
    if pipeline.boolean_step("is there a limit buy order exists already?")[0]:
        pipeline.if_step(
            "is there current limit buy price lower than current price of TQQQ -5%?",
            then_steps=[
                "cancel limit buy order for TQQQ",
                "set limit buy order for TQQQ for price 3 percent below the current price"
            ],
            events=history_with_price)
    else:
        pipeline.step(
            "set limit buy order for TQQQ for price 3 percent below the current price.", 
            events=history_with_price)
summary, _ = pipeline.summarize_full_history()
print(summary)
