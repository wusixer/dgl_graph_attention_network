# .PHONY: fullseq sasa

format:
	isort .
	black .


# after installing the package run Make fullseq, sasa and graph to generate data to load
fullseq:
	python scripts/ghesquire-fullseq.py
#sasa values were generated for each atom, each residue, each chain, each molecule. Only used sasa value for residue level currently
sasa:
	python scripts/ghesquire-sasa.py

# add fluctuation 
fluc:
	python scripts/ghesquire-fluc.py	

# the files after ":" are the files dependent on this make command, it tells Make if the file changes, what files to look for
graphs: data/ghesquire_2011/sasa.pkl data/ghesquire_2011/protein_sequences.fasta
	python scripts/ghesquire-graphs.py


mlsubset: data/ghesquire_2011/graphs.pkl
	python scripts/ghesquire-experiment.py
