trajectories:
	python generate_trajectories.py spatial 100

colocations:
	python identify_colocation.py 30 False

cinch-trajectories:
	cp ../mesa-examples/examples/cinch-data-generator/CINCH5/singleruns/agent_traj* ./data/
	cp ../mesa-examples/examples/cinch-data-generator/CINCH5/singleruns/door_stats* ./data/
