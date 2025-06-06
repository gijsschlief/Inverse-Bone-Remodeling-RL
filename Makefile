install:
	pip install -r requirements.txt

test:
	pytest tests/

train-surrogate:
	python main.py train_surrogate --epochs=100

train-rl:
	python main.py train_rl --timesteps=200000
