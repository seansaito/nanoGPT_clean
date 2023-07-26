# Makefile

# Run this when adding new pip packages
create-requirements:
	pip freeze > requirements.all.txt && grep -Fvxf requirements.dev.txt requirements.all.txt > requirements.txt && rm requirements.all.txt


# Run preprocessing of Grant data
prepare-data:
	python -m src.pipelines.pipeline_prepare_data

# Train the model
train:
	python -m src.pipelines.pipeline_train

.PHONY: create-requirements prepare-data train
