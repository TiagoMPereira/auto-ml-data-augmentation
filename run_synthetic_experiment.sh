#!/bin/bash

# Datasets
# - binary:
#   - 37        diabetes                        768x9x2
#   - 44        spambase            		    4601x58x2
#   - 1462      banknote-authentication         1372x6x2
#   - 1479      hill-valley         		    1212x101x2
#   - 1510      wdbc                		    569x31x2
# - multiclass:
#   - 23        contraceptive-method-choice	    1473x10x10
#   - 181       yeast				            1484x9x10
#   - 1466      cardiotocography         	    2126x24x10
#   - 40691     wine-quality        		    1599x12x6
#   - 40975     car      		                1728x7x4
# - multilabel:
#   - 41465        emotions             	    593x78x6
#   - 41468        image                 	    2000x140x5
#   - 41470        reuters                 	    2000x250x7
#   - 41471        scene                 	    2407x300x6
#   - 41473        yeast                 	    2417x117x14

# datasets=(37 44 1462 1479 1510 23 181 1466 40691 40975 41465 41468 41470 41471 41473)
datasets=(37)
datasets=(synthetic_dataset2)
synthesizers=(none copulagan ctgan fastml gaussiancopula tvae)
synthesizer_thresh=(none 1 5 10 15 20)

echo Script execution started at $(date).

# Preparation
echo ======== Preparation ========
echo Started cleaning files from previous executions at $(date).
rm -rf __pycache* &> /dev/null
rm -rf Autogluon* &> /dev/null
rf -rf gama_* &> /dev/null
rm -rf structured* &> /dev/null
# rm -rf venv-* &> /dev/null
rm *.log &> /dev/null
# rm results/* &> /dev/null
echo Finished cleaning files from previous executions at $(date).

for id in ${datasets[@]}; do

    for synt in ${synthesizers[@]}; do
        for thresh in ${synthesizer_thresh[@]}; do

            echo ======== Processing ========
            echo Started processing dataset $id with synthesizer $synt and thresh $thresh at $(date).

            # # AutoGluon
            # echo ======== AutoGluon ========
            # # python3 -m venv venv-autogluon
            # source ./venv-autogluon/bin/activate
            # # python3 -m pip install --upgrade pip
            # # python3 -m pip install --upgrade setuptools pytictoc wheel torch==1.12+cpu torchvision==0.13.0+cpu torchtext==0.13.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html autogluon sdv==1.2.0
            # python3 ./automl_autogluon.py $id $synt $thresh

            # # AutoKeras
            # echo ======== AutoKeras ========
            # # python3 -m venv venv-autokeras
            # source ./venv-autokeras/bin/activate
            # # python3 -m pip install --upgrade pip
            # # python3 -m pip install --upgrade setuptools pytictoc wheel git+https://github.com/keras-team/keras-tuner.git scikit-learn autokeras sdv==1.2.0
            # python3 ./automl_autokeras.py $id $synt $thresh

            # AutoPyTorch
            echo ======== AutoPyTorch ========
            python3 -m venv venv-autopytorch
            source ./venv-autopytorch/bin/activate
            python3 -m pip install --upgrade pip
            python3 -m pip install --upgrade setuptools pytictoc wheel swig torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu autoPyTorch sdv==1.2.0
            python3 ./automl_autopytorch.py $id $synt $thresh

            # # AutoSklearn
            # echo ======== AutoSklearn ========
            # # python3 -m venv venv-autosklearn
            # source ./venv-autosklearn/bin/activate
            # python3 -m pip install --upgrade pip
            # python3 -m pip install --upgrade setuptools pytictoc wheel auto-sklearn sdv==1.2.0
            # python3 ./automl_autosklearn.py $id $synt $thresh

            # # EvalML
            # echo ======== EvalML ========
            # # python3 -m venv venv-evalml
            # source ./venv-evalml/bin/activate
            # # python3 -m pip install --upgrade pip
            # # python3 -m pip install --upgrade setuptools pytictoc wheel evalml sdv==1.2.0
            # python3 ./automl_evalml.py $id

            # # FLAML
            # echo ======== FLAML ========
            # # python3 -m venv venv-flaml
            # source ./venv-flaml/bin/activate
            # # python3 -m pip install --upgrade pip
            # # python3 -m pip install --upgrade setuptools pytictoc wheel flaml sdv==1.2.0
            # python3 ./automl_flaml.py $id

            # # GAMA
            # echo ======== GAMA ========
            # # python3 -m venv venv-gama
            # source ./venv-gama/bin/activate
            # # python3 -m pip install --upgrade pip
            # # python3 -m pip install --upgrade setuptools pytictoc wheel gama sdv==1.2.0
            # python3 ./automl_gama.py $id

            # # H2O
            # echo ======== H2O ========
            # # python3 -m venv venv-h2o
            # source ./venv-h2o/bin/activate
            # # python3 -m pip install --upgrade pip
            # # python3 -m pip install --upgrade setuptools pytictoc wheel requests tabulate future scikit-learn pandas h2o sdv==1.2.0
            # python3 ./automl_h2o.py $id

            # # LightAutoML
            # echo ======== LightAutoML ========
            # # python3 -m venv venv-lightautoml
            # source ./venv-lightautoml/bin/activate
            # # python3 -m pip install --upgrade pip
            # # python3 -m pip install --upgrade setuptools pytictoc wheel lightautoml sdv==1.2.0
            # python3 ./automl_lightautoml.py $id

            # # TPOT
            # echo ======== TPOT ========
            # # python3 -m venv venv-tpot
            # source ./venv-tpot/bin/activate
            # # python3 -m pip install --upgrade pip
            # # python3 -m pip install --upgrade setuptools pytictoc wheel deap update_checker tqdm stopit xgboost torch tpot sdv==1.2.0
            # python3 ./automl_tpot.py $id

            # # Plotter
            # echo ======== Plotter ========
            # # python3 -m venv venv-plotter
            # source ./venv-plotter/bin/activate
            # # python3 -m pip install --upgrade pip
            # # python3 -m pip install --upgrade setuptools pytictoc wheel scikit-learn matplotlib pandas tabulate openpyxl sdv==1.2.0
            # python3 ./plotter.py $id

            echo Finished processing dataset $id at $(date).
        done
    done

done

echo Script execution finished at $(date).
