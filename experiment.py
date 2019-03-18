from main import main

seeds = [42, 43, 44, 45, 46]

# Test effect of max length
for s in seeds:
    for l in range(1, 20):
        max_length_params = ["--seed", str(s), "--max-length", str(l)]
        main(max_length_params)
