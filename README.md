# DEGage2.0
<img width="3232" height="689" alt="workflow_figure" src="https://github.com/user-attachments/assets/95fa03a4-6a8e-4e47-a343-a8b0f7a2fb80" />
The DEGage2.0 package allows for differential expression analysis of bulk RNA-seq data. Through the novel statistical model, DOTNB, the estimated difference between two negative binomial distributions would allow for the probabilistic determination of differentially expressed genes (DEGs). 

# Installation 
To install DEGage2.0 run the following command: 
```
pip install https://github.com/SabrinaChunn/DEGage2.git
```

# Dependencies 
To run DEGage2.0, the following libraries are required for installation: 
```
numpy>=1.22.0
pandas>=1.5.0
scipy>=1.9.0
statsmodels>=0.13.0
sympy>=1.10.0
```

# DEGage2.0 Function

**DEGage2_0**  
count_file -> The file location of the Bulk RNA-seq count data  
groups -> A numpy array containing 0s and 1s corresponding to the group identity of each column  
optimiser_file -> The file location of the optimized threshold combinations  
DOTNB_thresh -> The threshold for filtering of the DOTNB test  
perm_thresh -> The threshold for filtering of the permutation test  
num_permutations -> The amount of permutations iterated within the permutation test  

```
def DEGage2_0(count_file,
              groups,
              optimizer_file = "NA",
              DOTNB_thresh = -1,
              perm_thresh = -1,
              num_permutations = 1000):

```

# Example Usage 
This sections provides samples usage of DEGage2.0 through the use of sample test files. 
```
import numpy as np
import DEGage2

# Define numpy array corresponding to group sizes
groups = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Define input and optimization file
count_file = r"C:\LocationOfFile\data\simulated_data_10x10.csv.csv"
optimization_file = r"C:\LocationOfFile\data\optimal_thresholds.csv"

# Call DEGage2_0
results_df = DEGage2.DEGage2_0(count_file, groups, optimizer_file = optimization_file)

# Display final results
print("DEGs identified: \n", results_df)
```

# Citation  
If you use the code, please cite the following paper:

Chunn, S. J., Petrany, A., & Chen, Y. (2026). DEGage2.0: Model-based Comparative Analysis of Bulk RNA-seq Datasets. To_Be_Determined.
