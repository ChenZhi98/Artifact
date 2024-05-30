from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics
from arguments import TrainingArguments
from transformers import AutoModelForCausalLM, HfArgumentParser



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
    # Multiply accuracy of each client by number of examples used
    perplexities = [num_examples * m["perplexity"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"perplexity": sum(perplexities) / sum(examples)}


def get_parameters(model):
    print("Get initial_parameters")
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

if __name__ == "__main__":
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base_model_dir)
    params = get_parameters(model)

    # Define strategy
    strategy = fl.server.strategy.FedYogi(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients =3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=fit_config,  # Pass the fit_config function
        on_evaluate_config_fn=evaluate_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(params),
    )

    # Start server
    fl.server.start_server(
        server_address="localhost:8888",
        config=fl.server.ServerConfig(num_rounds=10, round_timeout=7200),
        grpc_max_message_length=-1,
        strategy=strategy,
    )
