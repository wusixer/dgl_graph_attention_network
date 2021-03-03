# .PHONY: fullseq sasa

format:
	isort .
	black .



fullseq:
	python scripts/ghesquire-fullseq.py

sasa:
	python scripts/ghesquire-sasa.py

graphs: data/ghesquire_2011/sasa.pkl data/ghesquire_2011/protein_sequences.fasta
	python scripts/ghesquire-graphs.py


mlsubset: data/ghesquire_2011/graphs.pkl
	python scripts/ghesquire-experiment.py
