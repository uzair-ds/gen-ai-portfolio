from graph.graph_builder import build_graph

if __name__ == "__main__":
    app = build_graph()

    state = {
        "user_query": "Join customers and orders and clean data",
        "data_source": "data/sample.csv",
        "schema": {}
    }

    result = app.invoke(state)
    print(result["answer"])
