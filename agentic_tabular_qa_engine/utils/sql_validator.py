def validate_sql_plan(plan: dict, schema: dict):
    forbidden = ["delete", "update", "drop"]

    for value in plan.values():
        if any(f in str(value).lower() for f in forbidden):
            raise ValueError("Unsafe SQL detected")

    return True
