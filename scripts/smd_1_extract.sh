<<<<<<< HEAD
source smd_0_push.sh

python extract_json.py \
	${SMDATA_DIR}/raw/${1} \
	${SMDATA_DIR}/json_raw/${1} \
	${2}
=======
python -m ddc.datasets.sm.extract \
	${SM_DATA_DIR}/*.zip \
	${SM_DATA_DIR}/json
>>>>>>> origin/master_v2
