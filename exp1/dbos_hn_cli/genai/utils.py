from rich.console import Console

console = Console()


def print_dic_keys(d: dict, prefix: str = "") -> None:
    """Utility to print nested dictionary keys for debugging.
    A dictionary can be nested in a structure of dictionaries or lists.
    """
    if isinstance(d, dict):
        for key, value in d.items():
            console.print(f"{prefix}{key}")
            print_dic_keys(value, prefix=prefix + "  ")
    elif isinstance(d, list):
        for i, item in enumerate(d):
            console.print(f"{prefix}[{i}]")
            print_dic_keys(item, prefix=prefix + "  ")
