trajectories:
	python generate_trajectories.py spatial 100

colocations_epr:
	python identify_colocation_epr.py 30 False

colocations_abm:
	python identify_colocation_abm.py

cinch-trajectories:
	cp ../mesa-examples/examples/cinch-data-generator/CINCH5/singleruns/agent_traj* ./data/
	cp ../mesa-examples/examples/cinch-data-generator/CINCH5/singleruns/door_stats* ./data/

test:
	pytest

test-cov:
	pytest --cov=colocationpy

format:
	ruff format

lint:
	ruff check --fix

lf: lint format
