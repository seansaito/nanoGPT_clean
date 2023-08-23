# Makefile
args = $(foreach a,$($(subst -,_,$1)_args),$(if $(value $a), --$a="$($a)"))

train_args = config_path quick_test wandb_project wandb_run_name

# Run this when adding new pip packages
create-requirements:
	pip freeze > requirements.all.txt && grep -Fvxf requirements.dev.txt requirements.all.txt > requirements.txt && rm requirements.all.txt


# Run preprocessing of training data
prepare-data:
	python -m src.pipelines.pipeline_prepare_data

# Train the model
# e.g. make train --wandb_project=nano_gpt_clean --wandb_run_name=run_1
train:
	python -m src.pipelines.pipeline_train $(call args,$@)

.PHONY: create-requirements prepare-data train
