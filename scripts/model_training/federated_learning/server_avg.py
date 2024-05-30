from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


def fit_config(server_round: int):
    config = {
        "server_round": server_round, 
        "local_epochs": 1
    }
    return config

def evaluate_config(server_round: int):
    config = {
        "server_round": server_round, 
    }
    return config

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    perplexities = [num_examples * m["perplexity"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"perplexity": sum(perplexities) / sum(examples)}


if __name__ == "__main__":
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients =3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=fit_config,  
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start server
    fl.server.start_server(
        server_address="localhost:8888",
        config=fl.server.ServerConfig(num_rounds=10, round_timeout=7200),
        grpc_max_message_length=-1,
        strategy=strategy,
    )
