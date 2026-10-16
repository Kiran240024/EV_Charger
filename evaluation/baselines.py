
def never_charge(state):
    """Request zero charging power."""
    return 0


def maximum_charge(state):
    """Request the maximum aggregate charging power."""
    return 11


def cheap_only(state, price_threshold):
    """
    Charge at maximum requested power when the current
    electricity price is at or below the threshold.
    """
    current_price = float(state[20])
    return 11 if current_price <= price_threshold else 0
