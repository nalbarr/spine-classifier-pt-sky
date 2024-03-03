NOTEBOOK_LOCAL=model/spine_binary_classifier_pytorch.ipynb
HOST=localhost:8888
NOTEBOOK_URL=http://localhost:8888/lab/tree/model/spine_binary_classifier_pytorch.ipynb

help:
	@echo make run
	@echo make run-ui
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

run:
	python train.py

run-ui:
	jupyter lab $(NOTEBOOK_LOCAL)

sky-aws-init:
	@echo ""

sky-check:
	sky check

sky-launch:
	sky launch -c spine infra/sky-spine-uv.yaml

sky-exec:
	sky exec spine  infra/sky-spine-uv.yaml

sky-status:
	sky status -a

sky-ssh:
	ssh spine

sky-logs:
	sky logs spine

sky-down:
	sky down spine

sky-spot-launch:
	sky spot launch -n spine infra/sky-spine-uv.yaml

