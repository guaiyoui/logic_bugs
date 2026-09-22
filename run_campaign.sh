#!/usr/bin/env bash
# AbsOracle exploration campaign: env-sweep / dml-image / data-homo
# across PG assert builds and DuckDB versions.
cd /home/user/work/db_safety_v1
DUCK113=/home/user/work/db_safety/venv_duckdb_113/bin/python
PGB=/home/user/work/db_safety/pgbld
mkdir -p results/absoracle logs

run_pg () {  # $1=oracle  $2=build  $3=cases  $4=seed
  echo "== $1 $2 =="
  python3 -m absoracle.$1 --engine pg --prefix "$PGB/$2" \
      --cases "$3" --seed "$4" --out "results/absoracle/$2" \
      2>>"logs/$1_$2.err" | tail -1
}

case "$1" in
env)
  for b in pg186_assert pgmaster_assert pg166_assert pg170_assert; do
    run_pg env_sweep "$b" "${2:-400}" "${3:-1}"
  done
  python3 -m absoracle.env_sweep --engine duck --cases "${2:-400}" \
      --seed "${3:-1}" --out results/absoracle/duck152
  ;;
dml)
  for b in pg186_assert pgmaster_assert pg166_assert pg170_assert pg160_assert; do
    run_pg dml_image "$b" "${2:-800}" "${3:-1}"
  done
  python3 -m absoracle.dml_image --engine duck --cases "${2:-800}" \
      --seed "${3:-1}" --out results/absoracle/duck152
  ;;
homo)
  for b in pg186_assert pgmaster_assert pg166_assert pg170_assert; do
    run_pg data_homo "$b" "${2:-1500}" "${3:-1}"
  done
  python3 -m absoracle.data_homo --engine duck --cases "${2:-1500}" \
      --seed "${3:-1}" --out results/absoracle/duck152
  ;;
eqcoh)
  for b in pg186_assert pgmaster_assert pg170_assert pg180_assert; do
    run_pg eqcoh "$b" "${2:-150}" "${3:-1}"
  done
  python3 -m absoracle.eqcoh --engine duck --cases "${2:-150}" \
      --seed "${3:-1}" --out results/absoracle/duck152
  ;;
duck113)
  $DUCK113 -m absoracle.data_homo --engine duck --cases "${2:-1500}" \
      --seed "${3:-1}" --out results/absoracle/duck113
  $DUCK113 -m absoracle.dml_image --engine duck --cases "${2:-800}" \
      --seed "${3:-1}" --out results/absoracle/duck113
  $DUCK113 -m absoracle.env_sweep --engine duck --cases "${2:-400}" \
      --seed "${3:-1}" --out results/absoracle/duck113
  $DUCK113 -m absoracle.eqcoh --engine duck --cases "${2:-150}" \
      --seed "${3:-1}" --out results/absoracle/duck113
  ;;
*) echo "usage: $0 {env|dml|homo|eqcoh|duck113} [cases] [seed]"; exit 1;;
esac
echo "DONE $1"
