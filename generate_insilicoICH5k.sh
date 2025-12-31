if [ $(generate --help | grep InSilicoLVO | wc -l) -eq 1 ]; then
    generate /projects01/didsr-aiml/brandon.nelson/rsna-aneurysm-synthetic-detection/synthetic-aneurysm-50k/synthetic-aneurysm-50k_study_plan.csv --parallel --chunk_size 2000 #--overwrite
else
    echo "Error: Invalid version of generate function present. Please follow the steps below to reinstall InSilicoLVO:"
    echo "1. pip uninstall InSilicoLVO -y"
    echo "2. pip install --upgrade git+https://github.com/DIDSR/PedSilicoLVO.git"
fi
