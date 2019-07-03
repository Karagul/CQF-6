# Replicates Excel-version from M1L2

import math
import numpy as np


def main():
    """

    Defines the option and model parameters. Print present value of option.
    :return: Present value of option.
    """
    # Option definition and time steps
    stock = 100
    volatility = 0.2
    interest_rate = 0.05
    strike = 100
    is_call = True
    expiration = 1.0
    is_eur = True
    time_steps = 4

    value = binomial_model(stock, volatility, interest_rate, strike, is_call, expiration, is_eur, time_steps)
    print(value)


def underlying_prices(time_steps: int, s0: float, u: float, v: float) -> np.ndarray:
    """

    Build tree of underlying prices.
    :param time_steps: Number of time steps.
    :param s0: Underlying price.
    :param u: Size of upward move in underlying price.
    :param v: Size of downward move in underlying price.
    :return: Tree of underlying prices.
    """

    underlying = np.zeros([time_steps + 1, time_steps + 1])
    for i in range(time_steps + 1):
        for j in range(i + 1):
            underlying[i, j] = s0 * (u ** (i - j) * v ** j)
    return underlying


def option_value(time_steps: int, s0: np.ndarray, is_call: bool,
                 is_european: bool, k: float, df: float, p: float) -> np.ndarray:
    """

    Calculate present value of option value.
    :param time_steps: Number of time steps.
    :param s0: Underlying price.
    :param is_call: Call option, if not the put option.
    :param is_european: European style exercise, if not then American style.
    :param k: Strike price.
    :param df: Discount factor, continuous compounding.
    :param p: Probability of up move in underlying price.
    :return: Tree of option prices.
    """

    option = np.zeros([time_steps + 1, time_steps + 1])
    kej = np.full(5, k)

    call = -1
    is_eur = -1
    if is_call:
        call = 1
    if is_european:
        is_eur = 1
    option[:, time_steps] = np.maximum(call * (s0[time_steps, :] - kej), np.zeros(time_steps + 1))

    # Recursively get present value of option
    for i in range(time_steps - 1, -1, -1):
        for j in range(0, i + 1):
            option[j, i] = df * (p * option[j, i + 1] + (1 - p) * option[j + 1, i + 1])
            payoff = s0[j, i] - k
            option[j, i] = np.maximum(option[j, i], is_eur * payoff)

    return option


def binomial_model(stock: float,
                   volatility: float,
                   interest_rate: float,
                   strike: float,
                   is_call: bool,
                   expiration: float,
                   is_european: bool,
                   time_steps: int) -> float:
    """

    Main function for binomial option model.
    :param stock: Underlying price today.
    :param volatility: Annualized volatility of underlying (decimal form).
    :param interest_rate: Annualized interest rate (decimal form).
    :param strike: Option strike.
    :param is_call: Call option, if not the put option.
    :param expiration: Time to option expiry (years).
    :param is_european: European style exercise, if not then American style.
    :param time_steps: Number of time steps in model.
    :return:
    """
    # Set model parameters.
    step_length = expiration / time_steps
    u = 1 + volatility * math.sqrt(step_length)
    v = 1 - volatility * math.sqrt(step_length)
    p = 0.5 + interest_rate * math.sqrt(step_length) / (2 * volatility)
    df = 1 / (1 + step_length * interest_rate)

    underlying = underlying_prices(time_steps, stock, u, v)
    option = option_value(time_steps, underlying, is_call, is_european, strike, df, p)

    return option[0, 0]


if __name__ == '__main__':
    main()
