SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help prepare host-build build-context plan verify-assets
help:
	@echo 'Opt-in TensorFold-main build preparation; no model is started or stopped.'
	@echo 'prepare host-build build-context plan verify-assets'
prepare:
	@python3 prepare.py source
host-build:
	@test -n "$(ZIG)" || { echo 'Set ZIG to the pinned Linux ARM64 toolchain directory'; exit 1; }
	@python3 prepare.py host-build --zig "$(ZIG)"
build-context:
	@python3 prepare.py context
plan:
	@python3 profile.py plan --config "$(CONFIG)"
verify-assets:
	@python3 profile.py verify-assets --config "$(CONFIG)"
