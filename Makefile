# Multi-camera surveillance system — common tasks.
# Run from the repo root. `make install` once, then `make api` and `make web`
# in two terminals (and optionally `make demo-clips && make demo-cameras`).

VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
API_URL ?= http://localhost:8000

.PHONY: install api web test demo-clips demo-cameras publisher clean-data help

help:
	@echo "make install       - create venv, install Python + web deps"
	@echo "make api           - run the FastAPI backend + supervisor (port 8000)"
	@echo "make web           - run the Next.js dashboard (port 3000)"
	@echo "make test          - run the API test suite"
	@echo "make demo-clips    - generate demo videos from the test face fixtures"
	@echo "make demo-cameras  - register two demo cameras (API must be running)"
	@echo "make publisher     - publish a local webcam as MJPEG (SOURCE=0 PORT=8090)"

install:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	cd web && npm install

api:
	$(VENV)/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000

web:
	cd web && NEXT_PUBLIC_API_URL=$(API_URL) npm run dev

test:
	$(PY) -m pytest tests/api/ -q

demo-clips:
	$(PY) scripts/make_demo_clip.py tests/api/fixtures/face_known_1.jpg demo/lobby.mp4
	$(PY) scripts/make_demo_clip.py tests/api/fixtures/face_known_2.jpg demo/lab.mp4

demo-cameras:
	curl -s -X POST $(API_URL)/cameras -H 'Content-Type: application/json' \
	  -d '{"name":"Lobby Cam","location":"Main Lobby","source":"demo/lobby.mp4","enabled":true}'; echo
	curl -s -X POST $(API_URL)/cameras -H 'Content-Type: application/json' \
	  -d '{"name":"Lab Cam","location":"Lab 2","source":"demo/lab.mp4","enabled":true}'; echo

publisher:
	$(PY) publisher.py --source $(or $(SOURCE),0) --port $(or $(PORT),8090)

clean-data:
	rm -rf api/data
