from baseline import baseline

seeds = [42, 43, 44, 45]

# Test effect of max length
for s in seeds:
    for l in range(1, 20):
        max_length_params = ["--seed", str(s), "--max-length", str(l)]
        baseline(max_length_params)
