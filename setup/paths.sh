# Source from Bash after activating roboicl-policy. Explicit overrides are preserved.
export ROBOICL_CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

# One place selects the two interpreters; no Conda environment variables to save.
if [[ -z "${ROBOICL_SIM_PYTHON:-}" || -z "${ROBOICL_POLICY_PYTHON:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "Activate Conda before sourcing setup/paths.sh, or set ROBOICL_SIM_PYTHON and ROBOICL_POLICY_PYTHON." >&2
        return 1
    fi
    _roboicl_conda_base="$(conda info --base)" || return 1
    export ROBOICL_SIM_PYTHON="${ROBOICL_SIM_PYTHON:-$_roboicl_conda_base/envs/roboicl-sim/bin/python}"
    export ROBOICL_POLICY_PYTHON="${ROBOICL_POLICY_PYTHON:-$_roboicl_conda_base/envs/roboicl-policy/bin/python}"
    unset _roboicl_conda_base
fi
export ROBOICL_SIM_PYTHON ROBOICL_POLICY_PYTHON

export ROBOICL_DATA_ROOT="${ROBOICL_DATA_ROOT:-$ROBOICL_CODE_ROOT/data}"
export ROBOICL_RESULTS_ROOT="${ROBOICL_RESULTS_ROOT:-$ROBOICL_CODE_ROOT/results}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROBOICL_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Preload only for the simulator child process, when the environment has this library.
_roboicl_sim_prefix="$(dirname -- "$(dirname -- "$ROBOICL_SIM_PYTHON")")"
if [[ -z "${ROBOICL_SIM_LD_PRELOAD:-}" && -f "$_roboicl_sim_prefix/lib/libstdc++.so.6" ]]; then
    export ROBOICL_SIM_LD_PRELOAD="$_roboicl_sim_prefix/lib/libstdc++.so.6"
fi
unset _roboicl_sim_prefix
