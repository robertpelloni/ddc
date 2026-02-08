<<<<<<< HEAD
source smd_0_push.sh

python analyze_json.py \
	${SMDATA_DIR}/json_filt/${1}.txt ${2}
=======
python -m ddc.datasets.sm.analyze \
	${SM_DATA_DIR}/json/all.txt \
	${2}
>>>>>>> origin/master_v2
