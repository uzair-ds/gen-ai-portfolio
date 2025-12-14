def route_intent(state):
    q = state["user_query"].lower()
    if "join" in q or "sql" in q:
        return "data_ops"
    return "qa"
