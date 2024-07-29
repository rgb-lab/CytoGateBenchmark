

MOUSE_LINEAGE_GATING_STRATEGY = {
    "unsup_T_cells": ["CD45+", ["CD3+", "CD45+"]],
    "unsup_DP_T_cells": ["unsup_T_cells", ["CD3+", "CD4+", "CD8+", "CD45+"]],
    "unsup_DN_T_cells": ["unsup_T_cells", ["CD3+", "CD4-", "CD8-", "CD45+"]],
    "unsup_CD4_T_cells": ["unsup_T_cells", ["CD3+", "CD4+", "CD8-", "CD45+"]],
    "unsup_CD8_T_cells": ["unsup_T_cells", ["CD3+", "CD4-", "CD8+", "CD45+"]],
    "unsup_Neutrophils": ["CD45+", ["CD45+", "Ly6G+", "Ly6C+", "CD11b+"]],
    "unsup_Monocytes": ["CD45+", ["CD45+", "Ly6C+", "Ly6G-", "CD11b+", "NK1.1-"]],
    "unsup_B_cells": ["CD45+", ["CD45+", "B220+"]],
    "unsup_NK_cells": ["CD45+", ["CD45+", "NK1.1+"]],
    "unsup_Eosinophils": ["CD45+", ["CD45+", "Siglec_F+", "Ly6G-"]] 
}

MOUSE_LINEAGE_GATE_MAPPING = {
    "T_cells": "unsup_T_cells",
    "DN_T_cells": "unsup_DN_T_cells",
    "DP_T_cells": "unsup_DP_T_cells",
    "CD4_T_cells": "unsup_CD4_T_cells",
    "CD8_T_cells": "unsup_CD8_T_cells",
    "NK_cells": "unsup_NK_cells",
    "Eosinophils": "unsup_Eosinophils",
    "Neutrophils": "unsup_Neutrophils",
    "B_cells": "unsup_B_cells",
    "Monocytes": "unsup_Monocytes"
}

GIESE_TA_GATING_STRATEGY = {
    "unsup_T_cells": ["Lymphocytes", ["CD3+"]],
   
    "unsup_CD4_T_cells": ["unsup_T_cells", ["CD3+", "CD4+", "CD8-"]],
    "unsup_CD8_T_cells": ["unsup_T_cells", ["CD3+", "CD4-", "CD8+"]],
    "unsup_DN_T_cells": ["unsup_T_cells", ["CD3+", "CD4-", "CD8-"]],
    "unsup_DP_T_cells": ["unsup_T_cells", ["CD3+", "CD4+", "CD8+"]],
    "unsup_Tfh": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CXCR5_(CD185)+", "CD45RA-"]],

    "unsup_CD4_CM": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CD197+", "CD45RA-"]],
    "unsup_CD4_EM": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CD197-", "CD45RA-"]],
    "unsup_CD4_TEMRA": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CD197-", "CD45RA+"]],
    "unsup_CD4_Naive": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CD197+", "CD45RA+"]],
    
    "unsup_CD4_CM_CD28+_CD57+": ["unsup_CD4_CM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28+", "CD57+"]],
    "unsup_CD4_CM_CD28+_CD57-": ["unsup_CD4_CM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28+", "CD57-"]],
    "unsup_CD4_CM_CD28-_CD57+": ["unsup_CD4_CM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28-", "CD57+"]],
    "unsup_CD4_CM_CD28-_CD57-": ["unsup_CD4_CM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28-", "CD57-"]],

    "unsup_CD4_CM_TH1": ["unsup_CD4_CM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD183+", "CD196-"]],
    "unsup_CD4_CM_TH2": ["unsup_CD4_CM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD183-", "CD196-"]],
    "unsup_CD4_CM_TH17": ["unsup_CD4_CM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD183-", "CD196+"]],
    "unsup_CD4_CM_TH1_TH17": ["unsup_CD4_CM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD183+", "CD196+"]],
    
    "unsup_CD4_EM_CD28+_CD57+": ["unsup_CD4_EM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28+", "CD57+"]],
    "unsup_CD4_EM_CD28+_CD57-": ["unsup_CD4_EM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28+", "CD57-"]],
    "unsup_CD4_EM_CD28-_CD57+": ["unsup_CD4_EM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28-", "CD57+"]],
    "unsup_CD4_EM_CD28-_CD57-": ["unsup_CD4_EM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28-", "CD57-"]],

    "unsup_CD4_EM_TH1": ["unsup_CD4_EM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD183+", "CD196-"]],
    "unsup_CD4_EM_TH2": ["unsup_CD4_EM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD183-", "CD196-"]],
    "unsup_CD4_EM_TH17": ["unsup_CD4_EM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD183-", "CD196+"]],
    "unsup_CD4_EM_TH1_TH17": ["unsup_CD4_EM", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD183+", "CD196+"]],
    
    "unsup_CD4_TEMRA_CD28+_CD57+": ["unsup_CD4_TEMRA", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28+", "CD57+"]],
    "unsup_CD4_TEMRA_CD28+_CD57-": ["unsup_CD4_TEMRA", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28+", "CD57-"]],
    "unsup_CD4_TEMRA_CD28-_CD57+": ["unsup_CD4_TEMRA", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28-", "CD57+"]],
    "unsup_CD4_TEMRA_CD28-_CD57-": ["unsup_CD4_TEMRA", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28-", "CD57-"]],
    
    "unsup_CD4_Naive_CD28+_CD57+": ["unsup_CD4_Naive", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28+", "CD57+"]],
    "unsup_CD4_Naive_CD28+_CD57-": ["unsup_CD4_Naive", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28+", "CD57-"]],
    "unsup_CD4_Naive_CD28-_CD57+": ["unsup_CD4_Naive", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28-", "CD57+"]],
    "unsup_CD4_Naive_CD28-_CD57-": ["unsup_CD4_Naive", ["CD3+", "CD4+", "CD197+", "CD45RA-", "CD28-", "CD57-"]],
    
    "unsup_CD8_CM": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA-"]],
    "unsup_CD8_EM": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197-", "CD45RA-"]],
    "unsup_CD8_TEMRA": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197-", "CD45RA+"]],
    "unsup_CD8_Naive": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA+"]],
    
    "unsup_CD8_CM_CD28+_CD57+": ["unsup_CD8_CM", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28+", "CD57+"]],
    "unsup_CD8_CM_CD28+_CD57-": ["unsup_CD8_CM", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28+", "CD57-"]],
    "unsup_CD8_CM_CD28-_CD57+": ["unsup_CD8_CM", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28-", "CD57+"]],
    "unsup_CD8_CM_CD28-_CD57-": ["unsup_CD8_CM", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28-", "CD57-"]],
    
    "unsup_CD8_EM_CD28+_CD57+": ["unsup_CD8_EM", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28+", "CD57+"]],
    "unsup_CD8_EM_CD28+_CD57-": ["unsup_CD8_EM", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28+", "CD57-"]],
    "unsup_CD8_EM_CD28-_CD57+": ["unsup_CD8_EM", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28-", "CD57+"]],
    "unsup_CD8_EM_CD28-_CD57-": ["unsup_CD8_EM", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28-", "CD57-"]],
    
    "unsup_CD8_TEMRA_CD28+_CD57+": ["unsup_CD8_TEMRA", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28+", "CD57+"]],
    "unsup_CD8_TEMRA_CD28+_CD57-": ["unsup_CD8_TEMRA", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28+", "CD57-"]],
    "unsup_CD8_TEMRA_CD28-_CD57+": ["unsup_CD8_TEMRA", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28-", "CD57+"]],
    "unsup_CD8_TEMRA_CD28-_CD57-": ["unsup_CD8_TEMRA", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28-", "CD57-"]],
    
    "unsup_CD8_Naive_CD28+_CD57+": ["unsup_CD8_Naive", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28+", "CD57+"]],
    "unsup_CD8_Naive_CD28+_CD57-": ["unsup_CD8_Naive", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28+", "CD57-"]],
    "unsup_CD8_Naive_CD28-_CD57+": ["unsup_CD8_Naive", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28-", "CD57+"]],
    "unsup_CD8_Naive_CD28-_CD57-": ["unsup_CD8_Naive", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28-", "CD57-"]],
    
    "unsup_CD8_CD38+_HLADR+": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD38+", "HLA_DR+"]],
    "unsup_CD8_CD38-_HLADR+": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD38-", "HLA_DR+"]],
    "unsup_CD8_CD38+_HLADR-": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD38+", "HLA_DR-"]],
    "unsup_CD8_CD38-_HLADR-": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD38-", "HLA_DR-"]],

    "unsup_CD8_CD28+_CD57+" : ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28+", "CD57+"]],
    "unsup_CD8_CD28-_CD57+": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28-", "CD57+"]],
    "unsup_CD8_CD28+_CD57-": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28+", "CD57-"]],
    "unsup_CD8_CD28-_CD57-": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA-", "CD28-", "CD57-"]],

    "unsup_CD8_ICOS+_PD1+": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA-", "ICOS_(CD278)+", "PD_1_(CD279)+"]],
    "unsup_CD8_ICOS-_PD1+": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA-", "ICOS_(CD278)-", "PD_1_(CD279)+"]],
    "unsup_CD8_ICOS+_PD1-": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA-", "ICOS_(CD278)+", "PD_1_(CD279)-"]],
    "unsup_CD8_ICOS-_PD1-": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD197+", "CD45RA-", "ICOS_(CD278)-", "PD_1_(CD279)-"]],

    "unsup_TCRgd": ["unsup_T_cells", ["CD3+", "TCR_g_d_APC_R700+"]],

    "unsup_TCRgd_CD28+_CD57+": ["unsup_TCRgd", ["CD3+", "TCR_g_d_APC_R700+", "CD28+", "CD57+"]],
    "unsup_TCRgd_CD28-_CD57+": ["unsup_TCRgd", ["CD3+", "TCR_g_d_APC_R700+", "CD28-", "CD57+"]],
    "unsup_TCRgd_CD28+_CD57-": ["unsup_TCRgd", ["CD3+", "TCR_g_d_APC_R700+", "CD28+", "CD57-"]],
    "unsup_TCRgd_CD28-_CD57-": ["unsup_TCRgd", ["CD3+", "TCR_g_d_APC_R700+", "CD28-", "CD57-"]],

    "unsup_TCRgd_CD45RA-_CD197+": ["unsup_TCRgd", ["CD3+", "CD4+", "CD197+", "CD45RA-"]],
    "unsup_TCRgd_CD45RA-_CD197-": ["unsup_TCRgd", ["CD3+", "CD4+", "CD197-", "CD45RA-"]],
    "unsup_TCRgd_CD45RA+_CD197-": ["unsup_TCRgd", ["CD3+", "CD4+", "CD197-", "CD45RA+"]],
    "unsup_TCRgd_CD45RA+_CD197+": ["unsup_TCRgd", ["CD3+", "CD4+", "CD197+", "CD45RA+"]],
    
}

GIESE_TA_GATE_MAP = {
    "root/Singlets/Lymphocytes":
            "root/Singlets/Lymphocytes",
    "root/Singlets/Lymphocytes/T_cells": 
            "root/Singlets/Lymphocytes/unsup_T_cells",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells",
    "root/Singlets/Lymphocytes/T_cells/DN_T-cells": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_DN_T_cells",
    "root/Singlets/Lymphocytes/T_cells/DP_T-cells": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_DP_T_cells",
    "root/Singlets/Lymphocytes/T_cells/TCR_gamma_delta": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_TCRgd",

    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_CM": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_CM",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_EM": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_EM",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_Naive": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_Naive",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_TEMRA": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_TEMRA",
    
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_CM/CM_TH1": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_CM/unsup_CD4_CM_TH1",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_CM/CM_TH2": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_CM/unsup_CD4_CM_TH2",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_CM/CM_TH1-TH17": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_CM/unsup_CD4_CM_TH1_TH17",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_CM/CM_TH17": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_CM/unsup_CD4_CM_TH17",

    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_CM/Q1:_CD28-_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_CM/unsup_CD4_CM_CD28-_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_CM/Q2:_CD28+_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_CM/unsup_CD4_CM_CD28+_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_CM/Q3:_CD28+_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_CM/unsup_CD4_CM_CD28+_CD57-",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_CM/Q4:_CD28-_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_CM/unsup_CD4_CM_CD28-_CD57-",
    
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_EM/EM_TH1": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_EM/unsup_CD4_EM_TH1",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_EM/EM_TH2": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_EM/unsup_CD4_EM_TH2",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_EM/EM_TH1-TH17": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_EM/unsup_CD4_EM_TH1_TH17",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_EM/EM_TH17": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_EM/unsup_CD4_EM_TH17",

    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_EM/Q1:_CD28-_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_EM/unsup_CD4_EM_CD28-_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_EM/Q2:_CD28+_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_EM/unsup_CD4_EM_CD28+_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_EM/Q3:_CD28+_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_EM/unsup_CD4_EM_CD28+_CD57-",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_EM/Q4:_CD28-_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_EM/unsup_CD4_EM_CD28-_CD57-",

    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_TEMRA/Q1:_CD28-_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_TEMRA/unsup_CD4_TEMRA_CD28-_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_TEMRA/Q2:_CD28+_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_TEMRA/unsup_CD4_TEMRA_CD28+_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_TEMRA/Q3:_CD28+_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_TEMRA/unsup_CD4_TEMRA_CD28+_CD57-",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_TEMRA/Q4:_CD28-_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_TEMRA/unsup_CD4_TEMRA_CD28-_CD57-",    
    
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_Naive/Q1:_CD28-_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_Naive/unsup_CD4_Naive_CD28-_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_Naive/Q2:_CD28+_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_Naive/unsup_CD4_Naive_CD28+_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_Naive/Q3:_CD28+_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_Naive/unsup_CD4_Naive_CD28+_CD57-",
    "root/Singlets/Lymphocytes/T_cells/CD4+_T-cells/CD4_Naive/Q4:_CD28-_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD4_T_cells/unsup_CD4_Naive/unsup_CD4_Naive_CD28-_CD57-",  
    
    
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_CM": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_CM",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_EM": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_EM",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_Naive": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_Naive",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_TEMRA": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_TEMRA",
    
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_Naive/Q1:_CD28-_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_Naive/unsup_CD8_Naive_CD28-_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_Naive/Q2:_CD28+_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_Naive/unsup_CD8_Naive_CD28+_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_Naive/Q3:_CD28+_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_Naive/unsup_CD8_Naive_CD28+_CD57-",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_Naive/Q4:_CD28-_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_Naive/unsup_CD8_Naive_CD28-_CD57-",

    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_EM/Q1:_CD28-_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_EM/unsup_CD8_EM_CD28-_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_EM/Q2:_CD28+_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_EM/unsup_CD8_EM_CD28+_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_EM/Q3:_CD28+_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_EM/unsup_CD8_EM_CD28+_CD57-",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_EM/Q4:_CD28-_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_EM/unsup_CD8_EM_CD28-_CD57-",

    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_TEMRA/Q1:_CD28-_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_TEMRA/unsup_CD8_TEMRA_CD28-_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_TEMRA/Q2:_CD28+_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_TEMRA/unsup_CD8_TEMRA_CD28+_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_TEMRA/Q3:_CD28+_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_TEMRA/unsup_CD8_TEMRA_CD28+_CD57-",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/CD8_TEMRA/Q4:_CD28-_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_TEMRA/unsup_CD8_TEMRA_CD28-_CD57-", 
 
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/Q1:_CD38-_,_HLA-DR+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_CD38-_HLADR+",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/Q2:_CD38+_,_HLA-DR+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_CD38+_HLADR+",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/Q3:_CD38+_,_HLA-DR-":
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_CD38+_HLADR-",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/Q4:_CD38-_,_HLA-DR-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_CD38-_HLADR-",

    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/Q5:_CD28-_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_CD28-_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/Q6:_CD28+_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_CD28+_CD57+",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/Q7:_CD28+_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_CD28+_CD57-",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/Q8:_CD28-_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_CD28-_CD57-",

    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/Q9:_ICOS_(CD278)-_,_PD-1_(CD279)+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_ICOS-_PD1+",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/Q10:_ICOS_(CD278)+_,_PD-1_(CD279)+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_ICOS+_PD1+",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/Q11:_ICOS_(CD278)+_,_PD-1_(CD279)-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_ICOS+_PD1-",
    "root/Singlets/Lymphocytes/T_cells/CD8+_T-cells/Q12:_ICOS_(CD278)-_,_PD-1_(CD279)-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_CD8_T_cells/unsup_CD8_ICOS-_PD1-",
  
    "root/Singlets/Lymphocytes/T_cells/TCR_gamma_delta/Q1:_CD28-_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_TCRgd/unsup_TCRgd_CD28-_CD57+",
    "root/Singlets/Lymphocytes/T_cells/TCR_gamma_delta/Q2:_CD28+_,_CD57+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_TCRgd/unsup_TCRgd_CD28+_CD57+",
    "root/Singlets/Lymphocytes/T_cells/TCR_gamma_delta/Q3:_CD28+_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_TCRgd/unsup_TCRgd_CD28+_CD57-",
    "root/Singlets/Lymphocytes/T_cells/TCR_gamma_delta/Q4:_CD28-_,_CD57-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_TCRgd/unsup_TCRgd_CD28-_CD57-",

    "root/Singlets/Lymphocytes/T_cells/TCR_gamma_delta/Q5:_CD197-_,_CD45RA+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_TCRgd/unsup_TCRgd_CD45RA+_CD197-",
    "root/Singlets/Lymphocytes/T_cells/TCR_gamma_delta/Q6:_CD197+_,_CD45RA+": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_TCRgd/unsup_TCRgd_CD45RA+_CD197+",
    "root/Singlets/Lymphocytes/T_cells/TCR_gamma_delta/Q7:_CD197+_,_CD45RA-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_TCRgd/unsup_TCRgd_CD45RA-_CD197+",
    "root/Singlets/Lymphocytes/T_cells/TCR_gamma_delta/Q8:_CD197-_,_CD45RA-": 
            "root/Singlets/Lymphocytes/unsup_T_cells/unsup_TCRgd/unsup_TCRgd_CD45RA-_CD197-",
    
}

ZPM_GATING_STRATEGY = {
    "unsup_T_cells": ["Lymphocytes", ["CD3+"]],
    "unsup_CD4_T_cells": ["unsup_T_cells", ["CD3+", "CD4+"]],
    "unsup_TH1_like": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CXCR3+", "CCR6-"]],
    "unsup_TH17_like": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CCR6+", "CXCR3-"]],
    "unsup_TH1_activated": ["unsup_TH1_like", ["HLA-DR+", "CD38+"]],
    "unsup_TH17_activated": ["unsup_TH17_like", ["HLA-DR+", "CD38+"]]

}

ZPM_GATE_MAPPING = {
    "CD3+": "unsup_T_cells",
    "CD4+": "unsup_CD4_T_cells",
    "TH1-like": "unsup_TH1_like",
    "TH17-like": "unsup_TH17_like",
    "TH1_activated": "unsup_TH1_activated",
    "TH17_activated": "unsup_TH17_activated"
}

HIMC_GATING_STRATEGY = {
    "unsup_T_cells": ["live", ["CD3+", "TCRgd-"]],
    "unsup_CD4_T_cells": ["unsup_T_cells", ["CD3+", "CD4+", "CD8-"]],
    "unsup_CD8_T_cells": ["unsup_T_cells", ["CD3+", "CD4-", "CD8+"]],
    "unsup_DN_T_cells": ["live", ["CD3+", "CD4-", "CD8-"]],
    "unsup_DP_T_cells": ["live", ["CD3+", "CD4+", "CD8+"]],
    "unsup_CD4_Naive": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CD45RA+", "CCR7+"]],
    "unsup_CD4_EM": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CD45RA-", "CCR7-"]],
    "unsup_CD4_CM": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CD45RA-", "CCR7+"]],
    "unsup_CD4_TEMRA": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CD45RA+", "CCR7-"]],
    "unsup_CD8_Naive": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD45RA+", "CCR7+"]],
    "unsup_CD8_EM": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD45RA-", "CCR7-"]],
    "unsup_CD8_CM": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD45RA-", "CCR7+"]],
    "unsup_CD8_TEMRA": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD45RA+", "CCR7-"]],
    "unsup_B_cells": ["live", ["CD19+", "CD20+", "CD3-"]],
    "unsup_IgD+_CD27-_B_cells": ["unsup_B_cells", ["IgD+", "CD27-"]],
    "unsup_IgD-_B_cells": ["unsup_B_cells", ["IgD-", "CD27-"]],
    "unsup_gdT_cells": ["live", ["CD3+", "TCRgd+"]],
    "unsup_NK_cells": ["live", ["CD3-", "CD16+", "CD56+"]],
    "unsup_DC": ["live", ["CD3-", "CD19-", "CD56-", "CD14-", "HLADR+"]],
}

HIMC_GATE_MAPPING = {
    "CD3+": "unsup_T_cells",
    "CD4_T_cells": "unsup_CD4_T_cells",
    "CD8+_T_cells": "unsup_CD8_T_cells",
    "DN_T_cells": "unsup_DN_T_cells",
    "DP_T_cells": "unsup_DP_T_cells",
    "CD4CM": "unsup_CD4_CM",
    "CD4EM": "unsup_CD4_EM",
    "CD4TEMRA": "unsup_CD4_TEMRA",
    "CD4Naive": "unsup_CD4_Naive",
    "CD8CM": "unsup_CD8_CM",
    "CD8EM": "unsup_CD8_EM",
    "CD8TEMRA": "unsup_CD8_TEMRA",
    "CD8Naive": "unsup_CD8_Naive",
    "B_cells": "unsup_B_cells",
    "NK_cells": "unsup_NK_cells",
    "DC": "unsup_DC",
    "IgD+_CD27+_B_cells": "unsup_IgD+_CD27-_B_cells",
    "IgD-_B_cells": "unsup_IgD-_B_cells",
    "TCRgd+": "unsup_gdT_cells"
}

OMIP_GATING_STRATEGY = {
    "unsup_T_cells": ["CD45+", ["CD3+", "TCRgd-"]],
    "unsup_NKT_like": ["unsup_T_cells", ["CD3+", "CD56+"]],
    "unsup_CD4_T_cells": ["unsup_T_cells", ["CD3+", "CD4+", "CD8-"]],
    "unsup_CD8_T_cells": ["unsup_T_cells", ["CD3+", "CD4-", "CD8+"]],
    "unsup_DN_T_cells": ["live", ["CD3+", "CD4-", "CD8-"]],
    "unsup_DP_T_cells": ["live", ["CD3+", "CD4+", "CD8+"]],
    "unsup_CD4_Naive": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CD45RA+", "CCR7+"]],
    "unsup_CD4_EM": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CD45RA-", "CCR7-"]],
    "unsup_CD4_CM": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CD45RA-", "CCR7+"]],
    "unsup_CD4_TEMRA": ["unsup_CD4_T_cells", ["CD3+", "CD4+", "CD45RA+", "CCR7-"]],
    "unsup_CD4_CM_TH1": ["unsup_CD4_CM", ["CD3+", "CD4+", "CD45RA-", "CCR7+", "CCR6-", "CXCR3+"]],
    "unsup_CD4_CM_TH17": ["unsup_CD4_CM", ["CD3+", "CD4+", "CD45RA-", "CCR7+", "CCR6+", "CXCR3-"]],
    "unsup_CD4_CM_TH1_TH17": ["unsup_CD4_CM", ["CD3+", "CD4+", "CD45RA-", "CCR7+", "CCR6+", "CXCR3+"]],
    "unsup_CD4_CM_TH2": ["unsup_CD4_CM", ["CD3+", "CD4+", "CD45RA-", "CCR7+", "CCR6-", "CXCR3-"]],
    "unsup_CD8_Naive": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD45RA+", "CCR7+"]],
    "unsup_CD8_EM": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD45RA-", "CCR7-"]],
    "unsup_CD8_CM": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD45RA-", "CCR7+"]],
    "unsup_CD8_TEMRA": ["unsup_CD8_T_cells", ["CD3+", "CD8+", "CD45RA+", "CCR7-"]],
    "unsup_B_cells": ["live", ["CD19+", "CD20+", "CD3-"]],
    "unsup_gdT_cells": ["live", ["CD3+", "TCRgd+"]],
    "unsup_NK_cells": ["live", ["CD3-", "CD16+", "CD56+"]],
    "unsup_CD11c_DC": ["live", ["CD3-", "CD19-", "CD56-", "CD14-", "HLADR+","CD11c+", "CD123-"]],
    "unsup_CD123_DC": ["live", ["CD3-", "CD19-", "CD56-", "CD14-", "HLADR+","CD123+", "CD11c-"]],
}

OMIP_GATE_MAPPING = {
    "CD3+_CD56-": "unsup_T_cells",
    "NKT_like_cells": "unsup_NKT_like",
    "CD4+_T_cells": "unsup_CD4_T_cells",
    "CD8+_T_cells": "unsup_CD8_T_cells",
    "DN_T_cells": "unsup_DN_T_cells",
    "DP_T_cells": "unsup_DP_T_cells",
    "CD4CM": "unsup_CD4_CM",
    "CD4EM": "unsup_CD4_EM",
    "CD4TEMRA": "unsup_CD4_TEMRA",
    "CD4Naive": "unsup_CD4_Naive",
    "CMTh1": "unsup_CD4_CM_TH1",
    "CMTh2": "unsup_CD4_CM_TH2",
    "CMTh17": "unsup_CD4_CM_TH17",
    "CMTh1Th17": "unsup_CD4_CM_TH1_TH17",
    "CD8CM": "unsup_CD8_CM",
    "CD8EM": "unsup_CD8_EM",
    "CD8TEMRA": "unsup_CD8_TEMRA",
    "Naive_CD8+": "unsup_CD8_Naive",
    "B_cells": "unsup_B_cells",
    "NK_cells": "unsup_NK_cells",
    "CD11c+_DC": "unsup_CD11c_DC",
    "CD123+_DC": "unsup_CD123_DC",
    "CD3+_TCRgd+": "unsup_gdT_cells",
}