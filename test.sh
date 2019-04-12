
# cultural/bio evolution: population size experiments
python main.py --population-size 4 & python main.py --population-size 8 & python main.py --evolution --population-size 4 & python main.py --evolution --population-size 8

python main.py --population-size 16 & python main.py --population-size 32 & python main.py --evolution --population-size 16 & python main.py --evolution --population-size 32

# testing different cultural sampling methods 
python main.py --culling-mode best & python main.py --culling-mode age

