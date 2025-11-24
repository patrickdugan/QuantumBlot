.PHONY: help install embed run full status clean test

# Default settings
INPUT ?= conversations.txt
THEME ?= 2
SHOTS ?= 8192
MODEL ?= e5
BACKEND ?= ibm_torino

help:
	@echo "🌊 QFT Runner - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install Node.js dependencies"
	@echo "  make install-python   Install Python dependencies"
	@echo ""
	@echo "Pipeline:"
	@echo "  make embed            Generate embeddings (INPUT=file, MODEL=e5|qwen)"
	@echo "  make run              Run QFT on existing vectors (THEME=2, SHOTS=8192)"
	@echo "  make full             Run full pipeline (INPUT=file, THEME=2)"
	@echo "  make status           Show pipeline status"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            Remove generated artifacts"
	@echo "  make clean-all        Remove all outputs and cache"
	@echo "  make test             Run test pipeline"
	@echo ""
	@echo "Examples:"
	@echo "  make embed INPUT=data.txt MODEL=e5"
	@echo "  make run THEME=5 SHOTS=16384 BACKEND=ibm_brisbane"
	@echo "  make full INPUT=conversations.txt THEME=3"

install:
	@echo "📦 Installing Node.js dependencies..."
	npm install

install-python:
	@echo "🐍 Installing Python dependencies..."
	pip install --break-system-packages numpy qiskit qiskit-ibm-runtime sentence-transformers requests

embed:
	@echo "📊 Generating embeddings..."
	@echo "   Input: $(INPUT)"
	@echo "   Model: $(MODEL)"
	node qft-runner.js embed --input $(INPUT) --model $(MODEL)

run:
	@echo "🌊 Running QFT..."
	@echo "   Theme: $(THEME)"
	@echo "   Shots: $(SHOTS)"
	@echo "   Backend: $(BACKEND)"
	@if [ ! -f "$(INPUT:.txt=_$(MODEL).npy)" ]; then \
		echo "❌ Vectors not found. Run 'make embed' first."; \
		exit 1; \
	fi
	node qft-runner.js run \
		--vectors $(INPUT:.txt=_$(MODEL).npy) \
		--theme-id $(THEME) \
		--shots $(SHOTS) \
		--backend $(BACKEND) \
		--layered

full:
	@echo "🚀 Running full pipeline..."
	@echo "   Input: $(INPUT)"
	@echo "   Theme: $(THEME)"
	@echo "   Shots: $(SHOTS)"
	@echo "   Model: $(MODEL)"
	node qft-runner.js full \
		--input $(INPUT) \
		--theme-id $(THEME) \
		--shots $(SHOTS) \
		--model $(MODEL) \
		--layered

status:
	@echo "📊 Pipeline Status:"
	@node qft-runner.js status

clean:
	@echo "🧹 Cleaning pipeline artifacts..."
	rm -f qft_counts.json
	rm -f decoded_evidence.json
	rm -f request_skeleton.json
	@echo "✅ Cleaned"

clean-all: clean
	@echo "🧹 Deep cleaning..."
	rm -f qft_Z.npy qft_mu.npy qft_W.npy
	rm -f vectors_pca_topk.npy
	rm -f *_e5.npy *_e5.jsonl
	rm -f *_qwen.npy *_qwen.jsonl
	rm -f conversations_bucketed.txt
	@echo "✅ All artifacts removed"

test:
	@echo "🧪 Running test pipeline..."
	@echo "Creating test data..."
	@echo "Test message 1: quantum interference patterns" > test.txt
	@echo "Test message 2: fourier transform analysis" >> test.txt
	@echo "Test message 3: embedding vector processing" >> test.txt
	@echo ""
	@make full INPUT=test.txt THEME=1 SHOTS=1024
	@echo ""
	@echo "✅ Test complete!"
	@make status

# Quick aliases
e: embed
r: run
f: full
s: status

# Development shortcuts
dev-embed:
	@make embed INPUT=conversations.txt MODEL=e5

dev-run:
	@make run THEME=2 SHOTS=4096

dev-full:
	@make full INPUT=conversations.txt THEME=2 SHOTS=4096

# Multi-theme analysis
themes:
	@echo "🎨 Running multi-theme analysis..."
	@for i in 0 1 2 3 4 5; do \
		echo ""; \
		echo "Theme $$i..."; \
		make run THEME=$$i SHOTS=8192; \
		mv qft_counts.json theme_$${i}_counts.json 2>/dev/null || true; \
	done
	@echo "✅ Multi-theme analysis complete"
	@ls -lh theme_*_counts.json

# Batch processing
batch:
	@echo "📦 Batch processing files..."
	@for file in *.txt; do \
		echo "Processing $$file..."; \
		make full INPUT=$$file THEME=2; \
		mkdir -p results/$${file%.txt}; \
		mv qft_counts.json results/$${file%.txt}/ 2>/dev/null || true; \
		mv decoded_evidence.json results/$${file%.txt}/ 2>/dev/null || true; \
	done
	@echo "✅ Batch complete - results in ./results/"

# Check environment
check-env:
	@echo "🔍 Checking environment..."
	@echo -n "IBM_CLOUD_API_KEY: "
	@if [ -z "$$IBM_CLOUD_API_KEY" ]; then echo "❌ Not set"; else echo "✅ Set"; fi
	@echo -n "IBM_QUANTUM_CRN: "
	@if [ -z "$$IBM_QUANTUM_CRN" ]; then echo "❌ Not set"; else echo "✅ Set"; fi
	@echo -n "Python: "
	@which python3 > /dev/null && echo "✅ $(shell python3 --version)" || echo "❌ Not found"
	@echo -n "Node.js: "
	@which node > /dev/null && echo "✅ $(shell node --version)" || echo "❌ Not found"
	@echo -n "qblot.env: "
	@if [ -f "qblot.env" ]; then echo "✅ Found"; else echo "❌ Missing"; fi

# Load environment from qblot.env
env:
	@if [ -f "qblot.env" ]; then \
		echo "Loading qblot.env..."; \
		export $$(cat qblot.env | grep -v '^#' | xargs); \
	else \
		echo "❌ qblot.env not found"; \
	fi

# Show logs
logs:
	@echo "📋 Recent logs:"
	@tail -n 50 nohup.out 2>/dev/null || echo "No logs found"

# Info
info:
	@echo "📊 QFT Pipeline Information"
	@echo ""
	@echo "Configuration:"
	@echo "  Backend: $(BACKEND)"
	@echo "  Default Shots: $(SHOTS)"
	@echo "  Default Model: $(MODEL)"
	@echo "  Target Dimension: 768"
	@echo "  Qubits: 17"
	@echo "  Sparsity: 70%"
	@echo ""
	@echo "Available Backends:"
	@echo "  - ibm_torino (127 qubits)"
	@echo "  - ibm_brisbane (133 qubits)"
	@echo "  - ibm_osaka (127 qubits)"
	@echo "  - ibm_kyoto (127 qubits)"
