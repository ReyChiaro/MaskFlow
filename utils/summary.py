import builtins
import torch.nn as nn


def convert_integer(num: int) -> str:
    if num >= 1e9:
        return f"{num / 1e9:>6.2f} B"
    elif num >= 1e6:
        return f"{num / 1e6:>6.2f} M"
    else:
        return f"{num:,}"


def summarize_model(model: nn.Module) -> dict[str, int | float]:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    return {
        "Num Params": convert_integer(total_params),
        "Num Trainable Params": convert_integer(trainable_params),
        "Model Size (MB)": f"{size_all_mb:.4f}",
        "Model Size (GB)": f"{size_all_mb / 1024:.4f}",
        "Trainable (%)": f"{trainable_params/total_params*100:.2f}",
    }


def get_summary_table(summary_dict) -> str:
    if not summary_dict:
        return
    if not isinstance(next(iter(summary_dict)), dict):
        summary_dict = {"<Model Name>": summary_dict}

    first_model = list(summary_dict.keys())[0]
    columns = list(summary_dict[first_model].keys())

    col_widths = {col: len(col) for col in columns}
    name_width = max(len("Model Name"), max(len(name) for name in summary_dict.keys()))

    for model_info in summary_dict.values():
        for col in columns:
            col_widths[col] = max(col_widths[col], len(str(model_info[col])))

    header_str = f"{'Model Name':<{name_width}}"
    for col in columns:
        header_str += f" | {col.capitalize():>{col_widths[col]}}"

    table = ""
    table += "\n" + " Summary ".center(len(header_str), "=")
    table += "\n" + header_str
    table += "\n" + "-" * len(header_str)

    for name, info in summary_dict.items():
        row_str = f"{name:<{name_width}}"
        for col in columns:
            val = info[col]
            row_str += f" | {val:>{col_widths[col]}}"
        table += "\n" + row_str
    table += "\n" + "=" * len(header_str)
    return table
