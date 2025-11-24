#!/bin/bash
# setup.sh - Quick setup script for QFT Runner

set -e

echo "🌊 QFT Runner Setup"
echo "==================="
echo ""

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+ first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi
echo "✅ Node.js found: $(node --version)"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+ first."
    exit 1
fi
echo "✅ Python found: $(python3 --version)"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Please install npm first."
    exit 1
fi
echo "✅ npm found: $(npm --version)"

echo ""
echo "📦 Installing Node.js dependencies..."
npm install

echo ""
echo "🐍 Installing Python dependencies..."
pip3 install --break-system-packages numpy qiskit qiskit-ibm-runtime sentence-transformers requests 2>/dev/null || \
pip3 install numpy qiskit qiskit-ibm-runtime sentence-transformers requests

echo ""
echo "🔑 Setting up environment..."

# Check if qblot.env exists
if [ -f "qblot.env" ]; then
    echo "✅ qblot.env found"
else
    echo "⚠️  qblot.env not found. Creating template..."
    cat > qblot.env << 'EOF'
export IBM_CLOUD_API_KEY="your_api_key_here"
export IBM_QUANTUM_CRN="crn:v1:bluemix:public:quantum-computing:us-east:..."
export QISKIT_IBM_RUNTIME_INSTANCE="crn:v1:bluemix:public:quantum-computing:us-east:..."
DEFAULT_BACKEND=ibm_torino
DEFAULT_SHOTS=8000
EOF
    echo "📝 Created qblot.env template - please edit with your credentials"
fi

# Make scripts executable
chmod +x qft-runner.js 2>/dev/null || true
chmod +x setup.sh 2>/dev/null || true

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit qblot.env with your IBM Quantum credentials"
echo "2. Source the environment: source qblot.env"
echo "3. Run a test: make test"
echo ""
echo "Quick start:"
echo "  node qft-runner.js status"
echo "  make help"
echo ""
echo "Documentation: See README.md"
