## 2 way ANOVA
import pandas 
from scipy import stats
import argparse
import os,sys,inspect
import numpy as np


'''

 Read file into pandas dataframe using args in command line. 


 The provided csv_file columns should be arranged as follows: 
 
 F1L1_F2L1_R1\t...\tF1L1_F2L1_Rn\tF1L1_F2L2_R1\t... F1L1_F2L2_Rn    ... F1Ln1_F2L1_R1 ... F1Ln1_F2L1_Rn F1Ln1_F2Ln2_R1  ... F1Ln1_F2Ln2_Rn
 
 (F: Factor; L: Level; R: Replicate; n1: number of levels for F1; n2: number of levels for F2; n: number of replicates/observations)
 
 Example:
 
 Factor 1 is Cell_Line, has 2 levels: Melanoma, Colon
 
 Factor 2 is Drug, has 3 levels: CT0, DTX, 5FU
 
 There are 3 replicates/observations per each cell line x Drug combination 
 
 The file columns should be:
 
 Mel_CT0_1  Mel_CT0_2   Mel_CT0_3   Mel_DTX_1   Mel_DTX_2   Mel_DTX_3   Mel_5FU_1   Mel_5FU_2   Mel_5FU_3 Col_CT0_1 Col_CT0_2   Col_CT0_3   Col_DTX_1   Col_DTX_2   Col_DTX_3   Col_5FU_1   Col_5FU_2   Col_5FU_3
 
'''



parser = argparse.ArgumentParser()
parser.add_argument("--csv_file", required=True, help="The csv file with factors as columns and independent measurements (peptides) as rows")
parser.add_argument("--num_cat_f1", required=True, type=int, help="The number of levels/categories for factor 1")
parser.add_argument("--num_cat_f2", required=True, type=int, help="The number of levels/categories for factor 2")
parser.add_argument("--num_repl", required=True, type=int, help="The number of observations per each factor 1 x factor 2 combination")
args = parser.parse_args()
datafile = args.csv_file
datafile_basename = os.path.basename(datafile).split(".")[0]
if not os.path.isfile(datafile):
    print "\nFile not found: %s " % datafile + "\n"
num_cat_f1 = args.num_cat_f1
num_cat_f2 = args.num_cat_f2
num_repl = args.num_repl


data = pandas.read_csv(datafile)



def dof(peptide_df):
    #### Calculate Degrees of Freedom
    N = len(peptide_df.dependent_variable)
    dof_factor_1 = len(peptide_df.factor_1.unique()) - 1
    dof_factor_2 = len(peptide_df.factor_2.unique()) - 1
    dof_factor_1_x_factor_2 = dof_factor_1*dof_factor_2 #interaction
    dof_w = N - (len(peptide_df.factor_1.unique())*len(peptide_df.factor_2.unique())) #within
    return(dof_factor_1, dof_factor_2, dof_factor_1_x_factor_2, dof_w)



def ssq(data, grand_mean):
    #SUM of SQUARES FOR EACH FACTOR (Independent variables: factor_1 AND factor_2): 
    ssq_factor_1 = sum( [(data[data.factor_1 == f1_category].dependent_variable.mean() - grand_mean)**2 for f1_category in data.factor_1] )
    ssq_factor_2 = sum( [(data[data.factor_2 == f2_category].dependent_variable.mean() - grand_mean)**2 for f2_category in data.factor_2] )
    #TOTAL SUM OF SQUARES:
    ssq_t = sum((data.dependent_variable - grand_mean)**2)

    f1_dict = {f1_lvl: data[data.factor_1 == f1_lvl] for f1_lvl in data.factor_1.unique()}
    
    #For within group variation I need first to estimate means in each treatment group (treatment = each combination of the two factors, f1xf2_dict)
    f1xf2_dict = { k: [v[v.factor_2 == f2_c].dependent_variable.mean() for f2_c in v.factor_2] for k, v in f1_dict.iteritems()}

    #And GET SSQ_w
    ssq_w = 0
    for k in f1_dict.keys():
        ssq_w += sum((f1_dict[k].dependent_variable - f1xf2_dict[k])**2)

    #SUM OF SQUARES OF THE INTERACTION 
    ssq_factor_1_x_factor_2 = ssq_t-ssq_factor_1-ssq_factor_2-ssq_w
    return(ssq_factor_1, ssq_factor_2, ssq_t, ssq_w, ssq_factor_1_x_factor_2)



def msq(ssq_factor_1, ssq_factor_2, ssq_factor_1_x_factor_2, ssq_w, dof_factor_1, dof_factor_2, dof_factor_1_x_factor_2, dof_w):
    #### Calculate Mean Squares:
    msq_factor_1 = ssq_factor_1 / dof_factor_1
    msq_factor_2 = ssq_factor_2 / dof_factor_2
    msq_factor_1_x_factor_2 = ssq_factor_1_x_factor_2 / dof_factor_1_x_factor_2 #INTERACTION
    msq_w = ssq_w/dof_w #WITHIN
    return(msq_factor_1, msq_factor_2, msq_factor_1_x_factor_2, msq_w)


def f_ratio(msq_factor_1, msq_factor_2, msq_factor_1_x_factor_2, msq_w):
    #### Calculate F-ratio: 
    # The F-statistic is simply the mean square for each effect and the interaction divided by the mean square for within (error/residual).
    f_factor_1 = msq_factor_1/msq_w
    f_factor_2 = msq_factor_2/msq_w
    f_factor_1_x_factor_2 = msq_factor_1_x_factor_2/msq_w
    return(f_factor_1, f_factor_2, f_factor_1_x_factor_2)


print("\nEstimating p-values\n")


def main():

    for index, row in data.iterrows():
        #Get pandas df for each peptide with columns dependent_variable, factor_1 and factor_2:
        peptide_df = pandas.DataFrame(columns=["dependent_variable", "factor_1", "factor_2"])
        peptide_df.factor_1 = [ "f1_%s"%cat_f1 for cat_f1 in range(num_cat_f1) for cat_f2 in range(num_cat_f2) for rep in range(num_repl) ]
        peptide_df.factor_2 = [ "f2_%s"%cat_f2 for cat_f1 in range(num_cat_f1) for cat_f2 in range(num_cat_f2) for rep in range(num_repl) ]
        peptide_df.dependent_variable = row.values[1:]
        #get Degrees of Freedom
        dof_factor_1, dof_factor_2, dof_factor_1_x_factor_2, dof_w = dof(peptide_df)
        #get grand mean Using Pandas DataFrame method mean on the dependent variable only
        grand_mean = peptide_df.dependent_variable.mean() #which is simply the mean of all values
        #Calculate sum of squares
        ssq_factor_1, ssq_factor_2, ssq_t, ssq_w, ssq_factor_1_x_factor_2 = ssq(peptide_df, grand_mean)
        #Calculate mean squares
        msq_factor_1, msq_factor_2, msq_factor_1_x_factor_2, msq_w = msq(ssq_factor_1, ssq_factor_2, ssq_factor_1_x_factor_2, ssq_w, dof_factor_1, dof_factor_2, dof_factor_1_x_factor_2, dof_w)
        #Calculate F-ratio (F-satistic)
        f_factor_1, f_factor_2, f_factor_1_x_factor_2 = f_ratio(msq_factor_1, msq_factor_2, msq_factor_1_x_factor_2, msq_w)
        #OBTAIN p-values
        #scipy.stats method f.sf checks if the obtained F-ratios are above the critical value. 
        #Use F-value for each effect and interaction as well as the degrees of freedom for them, and the degree of freedom within.
        #Null Hypothesis 1:  H0: no difference in means among the groups in factor 1
        p_factor_1 = stats.f.sf(f_factor_1, dof_factor_1, dof_w)
        #Null Hypothesis 2:  H0: no difference in means among the groups in factor 2
        p_factor_2 = stats.f.sf(f_factor_2, dof_factor_2, dof_w)
        #Null Hypothesis 3:  H0: cell-lines do not interact with factor_2s in the response
        p_factor_1_x_factor_2 = stats.f.sf(f_factor_1_x_factor_2, dof_factor_1_x_factor_2, dof_w)
        #ADD rows to the original pandas df with the three pvalues:
        data.loc[index, 'p_factor_1'] = p_factor_1
        data.loc[index, 'p_factor_2'] = p_factor_2
        data.loc[index, 'p_factor_1_x_factor_2'] = p_factor_1_x_factor_2

        # Create new data set with the extra columns with the p-values

        out_file_name = datafile_basename + "_2wANOVA.csv"
        data.to_csv(out_file_name, index=False)


if __name__ == '__main__':
    main()