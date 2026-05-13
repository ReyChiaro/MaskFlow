


def print_m(model_dict):
    # 定义表头
    # Model 占 15，Raw Params 占 18，Human Readable 占 12，Size 占 10
    header = f"{'Model':<15} | {'Params (Raw)':>18} | {'Short':>10} | {'Trainable':>10} | {'Size':>10}"
    sep = "-" * len(header)
    
    print(header)
    print(sep)
    
    for name, m in model_dict.items():
        p_raw = m['num_params']
        p_short = format_params(p_raw)
        t_short = format_params(m['num_trainable'])
        size = m['size']
        
        # {p_raw:>18,} 负责打印带逗号的原始整数
        print(f"{name:<15} | {p_raw:>18,} | {p_short:>10} | {t_short:>10} | {size:>10}")

print_model_table(data)