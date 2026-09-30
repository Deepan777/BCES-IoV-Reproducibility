.PHONY: budget verify-env phase0-tests protocol-test smoke fetch-next-data acquire-v2xtraj-review acquire-v2xtraj-continuation review-v2xtraj build-manifests check-data-adequacy phase3 labels-observational phase4 labels-sumo phase5 all-tests clean-transient

budget:
	python scripts/00_check_budget.py

verify-env:
	python scripts/00_check_budget.py
	python scripts/01_verify_environment.py
	python scripts/00_check_budget.py

phase0-tests:
	python scripts/00_check_budget.py
	python -m pytest tests/unit tests/integration -m phase0 --junitxml=outputs/phase0/phase0_test_report.xml
	python scripts/00_check_budget.py

protocol-test:
	python scripts/00_check_budget.py --report outputs/phase1/budget_pre.json
	python -m pytest tests/unit tests/integration -m phase1 --junitxml=outputs/phase1/protocol_geometry_test_report.xml
	python scripts/00_check_budget.py --report outputs/phase1/budget_post.json
	python scripts/phase1_gate.py

smoke:
	python scripts/00_check_budget.py --report outputs/phase2/budget_pre.json
	python scripts/02_smoke_pipeline.py
	python -m pytest tests/unit tests/integration tests/smoke -m phase2 --junitxml=outputs/phase2/smoke_test_report.xml
	python scripts/00_check_budget.py --report outputs/phase2/budget_post.json
	python scripts/phase2_gate.py

fetch-next-data:
	python scripts/00_check_budget.py --report outputs/phase3/budget_pre.json
	python scripts/03_fetch_next_data.py
	python scripts/00_check_budget.py --report outputs/phase3/budget_post.json

acquire-v2xtraj-review:
	python scripts/00_check_budget.py --report outputs/phase3/budget_pre.json
	python scripts/03_probe_v2xtraj_archive.py
	python scripts/03_inventory_v2xtraj_intersections.py
	python scripts/03_plan_v2xtraj_tranche.py
	python scripts/03_acquire_v2xtraj_tranche.py
	python scripts/03_review_v2xtraj_windows.py
	python scripts/03_materialize_v2xtraj_catalog.py
	python scripts/03_build_multidataset_state.py
	python scripts/00_check_budget.py --report outputs/phase3/budget_post.json

review-v2xtraj:
	python scripts/03_review_v2xtraj_windows.py
	python scripts/03_materialize_v2xtraj_catalog.py
	python scripts/03_build_multidataset_state.py
	python scripts/03_check_data_adequacy.py

acquire-v2xtraj-continuation:
	python scripts/00_check_budget.py --report outputs/phase3/budget_pre.json
	python scripts/03_plan_v2xtraj_continuation.py
	@echo The planner writes the next numbered plan and review; invoke the acquisition explicitly after reviewing those paths.
	@echo This target intentionally stops before transfer and never loops unattended.

build-manifests:
	python scripts/03_build_multidataset_state.py

check-data-adequacy:
	python scripts/03_check_data_adequacy.py

phase3:
	python scripts/00_check_budget.py --report outputs/phase3/budget_pre.json
	python scripts/03_build_multidataset_state.py
	python scripts/03_fetch_next_data.py
	python scripts/03_build_multidataset_state.py
	python scripts/03_check_data_adequacy.py
	python -m pytest tests/unit tests/integration -m phase3 --junitxml=outputs/phase3/phase3_test_report.xml
	python scripts/00_check_budget.py --report outputs/phase3/budget_post.json
	python scripts/phase3_gate.py

labels-observational:
	python scripts/00_check_budget.py --report outputs/phase4/budget_pre.json
	python scripts/04_fit_drift_scales.py
	python scripts/04_generate_observational_labels_v2.py
	python scripts/04_audit_observational_boundaries.py
	python scripts/00_check_budget.py --report outputs/phase4/budget_post.json

phase4:
	python -m pytest tests/unit tests/integration -m phase4 --junitxml=outputs/phase4/phase4_test_report.xml
	python scripts/04_phase4_gate.py

labels-sumo:
	python scripts/00_check_budget.py --report outputs/phase5/budget_pre.json
	python scripts/05_generate_sumo_labels.py
	python scripts/00_check_budget.py --report outputs/phase5/budget_post.json

phase5:
	python -m pytest tests/unit tests/integration -m phase5 --junitxml=outputs/phase5/phase5_test_report.xml
	python scripts/05_phase5_gate.py

all-tests:
	python scripts/00_check_budget.py
	python -m pytest --junitxml=outputs/phase2/all_tests_report.xml
	python scripts/00_check_budget.py

clean-transient:
	python scripts/clean_transient.py
