from typing import Literal
import os

DATASETS = ["mouse_lineages_bm", "mouse_lineages_pb", "mouse_lineages_spl", "human_t_cells", "HIMC", "OMIP", "ZPM"]

TIME_MAP_UNSUP = {
    "mouse_lineages_bm": "48:00:00",
    "mouse_lineages_pb": "48:00:00",
    "mouse_lineages_spl": "48:00:00",
    "human_t_cells": "120:00:00",
    "OMIP": "120:00:00",
    "ZPM": "120:00:00",
    "HIMC": "120:00:00",
}

TIME_MAP_SUP = {
    "mouse_lineages_bm": "120:00:00",
    "mouse_lineages_pb": "120:00:00",
    "mouse_lineages_spl": "120:00:00",
    "human_t_cells": "120:00:00",
    "OMIP": "120:00:00",
    "ZPM": "120:00:00",
    "HIMC": "120:00:00",
}

TIME_MAP_SAMPLER = {
    "mouse_lineages_bm": "120:00:00",
    "mouse_lineages_pb": "120:00:00",
    "mouse_lineages_spl": "120:00:00",
    "human_t_cells": "120:00:00",
    "OMIP": "120:00:00",
    "ZPM": "120:00:00",
    "HIMC": "120:00:00",
}

RAM_MAP_UNSUP = {
    "mouse_lineages_bm": "32gb",
    "mouse_lineages_pb": "32gb",
    "mouse_lineages_spl": "32gb",
    "human_t_cells": "64gb",
    "OMIP": "64gb",
    "ZPM": "64gb",
    "HIMC": "64gb"
}

RAM_MAP_SUP = {
    "mouse_lineages_bm": {
        "clfcomp": "32gb",
        "hyper_RF": "32gb",
        "hyper_DT": "32gb",
        "hyper_ET": "32gb",
        "hyper_ETS": "32gb",
        "hyper_MLP": "64gb",
        "hyper_KNN": "32gb",
    },
    "mouse_lineages_pb": {
        "clfcomp": "128gb",
        "hyper_RF": "32gb",
        "hyper_DT": "32gb",
        "hyper_ET": "32gb",
        "hyper_ETS": "64gb",
        "hyper_MLP": "128gb", # hyperparameter tuning...
        "hyper_KNN": "32gb",
    },
    "mouse_lineages_spl": {
        "clfcomp": "32gb",
        "hyper_RF": "32gb",
        "hyper_DT": "32gb",
        "hyper_ET": "32gb",
        "hyper_ETS": "32gb",
        "hyper_MLP": "64gb",
        "hyper_KNN": "32gb",
    },
    "human_t_cells": {
        "clfcomp": "64gb",
        "hyper_RF": "64gb",
        "hyper_DT": "32gb",
        "hyper_ET": "32gb",
        "hyper_ETS": "128gb", # for hyperparameter tuning
        "hyper_MLP": "32gb",
        "hyper_KNN": "32gb",
    },
    "OMIP": {
        "clfcomp": "128b", # even with labelspreading n_jobs = 1...
        "hyper_RF": "32gb",
        "hyper_DT": "32gb",
        "hyper_ET": "32gb",
        "hyper_ETS": "128gb", # for hyperparameter tuning
        "hyper_MLP": "64gb",
        "hyper_KNN": "32gb",
    },
    "ZPM": {
        "clfcomp": "128gb",
        "hyper_RF": "32gb",
        "hyper_DT": "32gb",
        "hyper_ET": "32gb",
        "hyper_ETS": "32gb",
        "hyper_MLP": "64gb",
        "hyper_KNN": "32gb",
    },
    "HIMC": {
        "clfcomp": "32gb",
        "hyper_RF": "32gb",
        "hyper_DT": "32gb",
        "hyper_ET": "32gb",
        "hyper_ETS": "64gb",
        "hyper_MLP": "64gb",
        "hyper_KNN": "32gb",
    }
}

RAM_MAP_SAMPLER = {
    "mouse_lineages_bm": {
        "sampler_DT": "16gb",
        "sampler_RF": "16gb",
        "sampler_ET": "16gb",
        "sampler_ETS": "16gb",
        "sampler_MLP": "16gb",
        "sampler_KNN": "16gb",
    },
    "mouse_lineages_pb": {
        "sampler_DT": "32gb",
        "sampler_RF": "32gb",
        "sampler_ET": "32gb",
        "sampler_ETS": "32gb",
        "sampler_MLP": "32gb",
        "sampler_KNN": "32gb",
    },
    "mouse_lineages_spl": {
        "sampler_DT": "16gb",
        "sampler_RF": "16gb",
        "sampler_ET": "16gb",
        "sampler_ETS": "16gb",
        "sampler_MLP": "16gb",
        "sampler_KNN": "16gb",
    },
    "human_t_cells": {
        "sampler_DT": "32gb",
        "sampler_RF": "32gb",
        "sampler_ET": "32gb",
        "sampler_ETS": "32gb",
        "sampler_MLP": "32gb",
        "sampler_KNN": "32gb",
    },
    "OMIP": {
        "sampler_DT": "32gb",
        "sampler_RF": "32gb",
        "sampler_ET": "32gb",
        "sampler_ETS": "32gb",
        "sampler_MLP": "32gb",
        "sampler_KNN": "32gb",
    },
    "ZPM": {
        "sampler_DT": "32gb",
        "sampler_RF": "32gb",
        "sampler_ET": "32gb",
        "sampler_ETS": "32gb",
        "sampler_MLP": "32gb",
        "sampler_KNN": "32gb",
    },
    "HIMC": {
        "sampler_DT": "32gb",
        "sampler_RF": "32gb",
        "sampler_ET": "32gb",
        "sampler_ETS": "32gb",
        "sampler_MLP": "32gb",
        "sampler_KNN": "32gb",
    }
}

JOB_NAME_MAP_DATASET = {
    "mouse_lineages_bm": "MB",
    "mouse_lineages_pb": "MP",
    "mouse_lineages_spl": "MS",
    "HIMC": "HI",
    "human_t_cells": "GI",
    "OMIP": "OM",
    "ZPM": "ZP"
}

JOB_NAME_MAP_LEARNING_ENTITY = {
    "supervised": "S",
    "unsupervised": "U",
    "sampler": "P"
}

JOB_NAME_MAP_ASSAY = {
    "clfcomp": "CLF",
    "algcomp": "ALG",
    "train_size": "TRS",

    "hyper_RF": "HRF",
    "hyper_DT": "HDT",
    "hyper_ET": "HET",
    "hyper_ETS": "HES",
    "hyper_KNN": "KNN",
    "hyper_MLP": "HML",

    "sampler_RF": "SRF",
    "sampler_DT": "SDT",
    "sampler_ET": "SET",
    "sampler_ETS": "SES",
    "sampler_KNN": "SKN",
    "sampler_MLP": "SML",

    "leiden": "LEI",
    "parc": "PAR",
    "phenograph": "PHE",
    "flowsom": "FLS",
    "preproc": "PRE",
    "sens": "SNS"
}

def generate_scripts(output_dir: str,
                     learning_entity: Literal["supervised", "unsupervised", "sampler"],
                     dataset: str,
                     continue_analysis: bool = False):
    if dataset not in DATASETS:
        raise TypeError(f"Dataset not found. Please choose one of {DATASETS}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if learning_entity == "supervised":
        supervised_assays = ["clfcomp", "hyper_RF", "hyper_DT", "hyper_ET", "hyper_ETS", "hyper_MLP", "hyper_KNN"]
        for assay in supervised_assays:
            with open(os.path.join(output_dir, f"{dataset}_{assay}.sh"), "w") as script:
                script.writelines("#!/bin/bash\n#SBATCH --partition=single\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=16\n")
                script.writelines(f"#SBATCH --time={TIME_MAP_SUP[dataset]}\n#SBATCH --mem={RAM_MAP_SUP[dataset][assay]}\n")
                script.writelines(f"#SBATCH --output=0_{dataset}_{assay}.out\n#SBATCH --error=0_{dataset}_{assay}.err\n")
                script.writelines(f"#SBATCH --mail-user=slurmjobs.helix@gmail.com\n#SBATCH --mail-type=FAIL\n")
                script.writelines(f"#SBATCH --job-name={JOB_NAME_MAP_DATASET[dataset]}_{JOB_NAME_MAP_LEARNING_ENTITY['supervised']}_{JOB_NAME_MAP_ASSAY[assay]}\n")
                script.writelines(f"python {dataset}_{assay}.py")
            with open(os.path.join(output_dir, f"{dataset}_{assay}.py"), "w") as script:
                script.writelines(f"import os\nos.chdir('../../')\nprint('Running {assay} on dataset {dataset}')\n")
                script.writelines(f"from FACSPyPaper.Figure2_supervised.{dataset}_{assay} import run\n")
                if continue_analysis:
                    script.writelines("run(continue_analysis = True)\n")
                else:
                    script.writelines("run()\n")
                script.writelines("print('Finished')\n")
        with open(os.path.join(output_dir, f"000_{learning_entity}_{dataset}.sh"), "w") as script:
            script.writelines("#!/bin/bash\n")
            for assay in supervised_assays:
                script.writelines(f"sbatch {dataset}_{assay}.sh\n")

    if learning_entity == "unsupervised":
        unsupervised_assays = ["algcomp", "leiden", "parc", "phenograph", "flowsom", "preproc", "sens"]
        for assay in unsupervised_assays:
            with open(os.path.join(output_dir, f"{dataset}_{assay}.sh"), "w") as script:
                script.writelines("#!/bin/bash\n#SBATCH --partition=single\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=16\n")
                script.writelines(f"#SBATCH --time={TIME_MAP_UNSUP[dataset]}\n#SBATCH --mem={RAM_MAP_UNSUP[dataset]}\n")
                script.writelines(f"#SBATCH --output=0_{dataset}_{assay}.out\n#SBATCH --error=0_{dataset}_{assay}.err\n")
                script.writelines(f"#SBATCH --mail-user=slurmjobs.helix@gmail.com\n#SBATCH --mail-type=FAIL\n")
                script.writelines(f"#SBATCH --job-name={JOB_NAME_MAP_DATASET[dataset]}_{JOB_NAME_MAP_LEARNING_ENTITY['unsupervised']}_{JOB_NAME_MAP_ASSAY[assay]}\n")
                script.writelines(f"python {dataset}_{assay}.py")
            with open(os.path.join(output_dir, f"{dataset}_{assay}.py"), "w") as script:
                script.writelines(f"import os\nos.chdir('../../')\nprint('Running {assay} on dataset {dataset}')\n")
                script.writelines(f"from FACSPyPaper.Figure3_unsupervised.{dataset}_{assay} import run\n")
                if continue_analysis:
                    script.writelines("run(continue_analysis = True)\n")
                else:
                    script.writelines("run()\n")
                script.writelines("print('Finished')\n")
        with open(os.path.join(output_dir, f"000_{learning_entity}_{dataset}.sh"), "w") as script:
            script.writelines("#!/bin/bash\n")
            for assay in unsupervised_assays:
                script.writelines(f"sbatch {dataset}_{assay}.sh\n")

    if learning_entity == "sampler":
        sampler_assays = ["sampler_RF", "sampler_DT", "sampler_ET", "sampler_ETS", "sampler_MLP", "sampler_KNN"]
        for assay in sampler_assays:
            with open(os.path.join(output_dir, f"{dataset}_{assay}.sh"), "w") as script:
                script.writelines("#!/bin/bash\n#SBATCH --partition=single\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=16\n")
                script.writelines(f"#SBATCH --time={TIME_MAP_SAMPLER[dataset]}\n#SBATCH --mem={RAM_MAP_SAMPLER[dataset][assay]}\n")
                script.writelines(f"#SBATCH --output=0_{dataset}_{assay}.out\n#SBATCH --error=0_{dataset}_{assay}.err\n")
                script.writelines(f"#SBATCH --mail-user=slurmjobs.helix@gmail.com\n#SBATCH --mail-type=FAIL\n")
                script.writelines(f"#SBATCH --job-name={JOB_NAME_MAP_DATASET[dataset]}_{JOB_NAME_MAP_LEARNING_ENTITY['sampler']}_{JOB_NAME_MAP_ASSAY[assay]}\n")
                script.writelines(f"python {dataset}_{assay}.py")
            with open(os.path.join(output_dir, f"{dataset}_{assay}.py"), "w") as script:
                script.writelines(f"import os\nos.chdir('../../')\nprint('Running {assay} on dataset {dataset}')\n")
                script.writelines(f"from FACSPyPaper.Figure4_sampler.{dataset}_{assay} import run\n")
                if continue_analysis:
                    script.writelines("run(continue_analysis = True)\n")
                else:
                    script.writelines("run()\n")
                script.writelines("print('Finished')\n")
        with open(os.path.join(output_dir, f"000_{learning_entity}_{dataset}.sh"), "w") as script:
            script.writelines("#!/bin/bash\n")
            for assay in sampler_assays:
                script.writelines(f"sbatch {dataset}_{assay}.sh\n")

    return




    

    
