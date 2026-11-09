if (( INSTALL )); then
  STAGE="install"
  if [[ ! -s "$SCRIPT_DIR/requirements.txt" ]]; then echo "requirements.txt is empty" | tee "$OUT/logs/01_install.log"; exit 3; fi
  set +e
  python -m pip install -r "$SCRIPT_DIR/requirements.txt" 2>&1 | tee "$OUT/logs/01_install.log"
  INSTALL_CODE=${PIPESTATUS[0]}
  set -e
  if (( INSTALL_CODE != 0 )); then
    if grep -qi "externally-managed-environment" "$OUT/logs/01_install.log"; then
      echo "externally-managed-environment detected; retrying with --break-system-packages" | tee -a "$OUT/logs/01_install.log"
      python -m pip install --break-system-packages -r "$SCRIPT_DIR/requirements.txt" 2>&1 | tee -a "$OUT/logs/01_install.log"
    else
      exit "$INSTALL_CODE"
    fi
  fi
fi
