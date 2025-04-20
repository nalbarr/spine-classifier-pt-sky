NOTEBOOK_LOCAL=model/spine_binary_classifier_pytorch.ipynb
HOST=localhost:8888
NOTEBOOK_URL=http://localhost:8888/lab/tree/model/spine_binary_classifier_pytorch.ipynb

help:
	@echo make run
	@echo make run-ui
	@echo ""
	@echo make install
	@echo ""
	@echo make sky-aws-init
	@echo ""
	@echo make sky check
	@echo ""
	@echo make sky-launch
	@echo make sky-exec
	@echo make sky-status
	@echo make sky-ssh
	@echo make sky-logs
	@echo make sky-down
	@echo ""
	@echo make sky-spot-launch
	@echo make sky-spot-logs

lint:
	flake8 train.py

format:
	black train.py

run:
	python train.py

run-ui:
	jupyter lab $(NOTEBOOK_LOCAL)

# https://docs.skypilot.co/en/latest/getting-started/installation.html
install:
	@echo NOTE: install uv
	@echo uv venv --seed --python 3.10
	@echo uv pip install "skypilot[kubernetes,aws]"

	@echo NOTE: uv sync
	@echo uv venv
	@echo source .venv/bin/activate
	@echo uv sync
	@echo uv run <script>

sky-aws-init:
	@echo NOTE: set AWS credentials!!!
	@echo "source ~/aws-utils/set-<aws_env>.sh"

sky-check:
	sky check

sky-launch:
	sky launch -c spine --cloud aws infra/sky-spine-uv.yaml

sky-exec:
	sky exec spine infra/sky-spine-uv.yaml

sky-status:
	sky status -a

sky-ssh:
	ssh spine

sky-logs:
	sky logs spine

sky-down:
	sky down spine

sky-spot-launch:
	sky launch -c spine -spot-instance infra/sky-spine-uv.yaml

