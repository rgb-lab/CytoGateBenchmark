from pathlib import Path
import itertools

def __generate_parameter_combinations(parameters: list[list]) -> list[list]:
    combinations = itertools.product(*parameters)
    return [",".join(ls) + "," for ls in combinations]

def _has_been_analyzed(file_path: Path,
                       parameters: list[list]):
    """
    Checks if all samples and all gates have been processed for the specific configuration
    Might be terribly inefficient...
    
    """
    combinations = __generate_parameter_combinations(parameters)
    with open(file_path, "r") as file:
        for combination in combinations:
            for line in file:
                if combination in line:
                    break
            else:
                print(f"The combination {combination} was not found... continuing analysis!")
                return False
            file.seek(0)
    return True