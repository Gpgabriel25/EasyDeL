#!/usr/bin/env bash
# =============================================================================
# EasyDeL + JAX 0.10 Bootstrap for TPU
# =============================================================================
# Installs EasyDeL (Gpgabriel25 fork) with JAX 0.10.1 and ejkernel 0.0.79+
# Handles the ejkernel/jax version mismatch by:
#   1. Installing EasyDeL normally (gets latest ejkernel)
#   2. If pip resolution fails due to jax version, force-install ejkernel
#   3. Final JAX 0.10.1 install overrides any downgrade
# =============================================================================
set -euo pipefail

PIP="pip install -q"
CONDA_BASE="${CONDA_BASE:-/tmp/miniconda3}"
export PATH="$CONDA_BASE/bin:$PATH"

echo "=== EasyDeL + JAX 0.10 Bootstrap ==="

# Step 1: Ensure JAX 0.10.1 is installed first (so it's not downgraded)
echo "[1/4] Installing JAX 0.10.1..."
$PIP 'jax[tpu]==0.10.1' 'jaxlib==0.10.1' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>/dev/null
python3 -c 'import jax; print(f"  JAX {jax.__version__}, devices: {jax.device_count()}")'

# Step 2: Clone and install EasyDeL fork (with ejkernel>=0.0.79 pin)
echo "[2/4] Installing EasyDeL (Gpgabriel25 fork)..."
EASDIR="${EASDIR:-/tmp/easydel}"
if [ ! -d "$EASDIR" ]; then
    git clone -q https://github.com/Gpgabriel25/EasyDeL.git "$EASDIR"
fi
cd "$EASDIR"

# Install EasyDeL — may downgrade JAX, we'll fix in step 4
$PIP -e . 2>/dev/null || echo "  (warnings OK — will fix JAX version next)"

# Step 3: Force ejkernel latest with --no-deps (skips jax version check)
echo "[3/4] Ensuring latest ejkernel..."
EJ_LATEST=$(pip index versions ejkernel 2>/dev/null | grep 'LATEST' | awk '{print $2}' || echo "0.0.79")
$PIP --no-deps "ejkernel==$EJ_LATEST" 2>/dev/null
python3 -c 'import ejkernel; print(f"  ejkernel {ejkernel.__version__}")'

# Step 4: Re-pin JAX 0.10.1 (EasyDeL may have downgraded to 0.9.x)
echo "[4/4] Re-pinning JAX 0.10.1..."
$PIP 'jax[tpu]==0.10.1' 'jaxlib==0.10.1' \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>/dev/null

# Final verification
echo ""
echo "=== Verification ==="
python3 -c "
import jax; import ejkernel; import easydel as ed
print(f'JAX:      {jax.__version__}')
print(f'ejkernel: {ejkernel.__version__}')
print(f'EasyDeL:  {ed.__version__}')
print(f'Devices:  {jax.device_count()}')
print(f'Mesh OK:  {len(jax.devices())} devices')
print('=== ALL GOOD ===')
"
echo ""
echo "Done! EasyDeL + JAX 0.10.1 + ejkernel $EJ_LATEST ready."
