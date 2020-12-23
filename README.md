# nsga-preference
Realization of Multi-object optimization for material design by preferred NSGA2.
This project use machine learning method of SVR to train a model, which can be used to predict the thermal expansion coefficient and thermal conductivity of ceramic coating materials.After that, multi-objective optimization was carried out by NSGA2-P.  
_test3.py_ is used to compare and select the basic model.  
_Ceramic coating dataset.xlsx_ is the original dataset.  
_hhh.txt_ is the dataset obtained by deleting the header from Ceramic coating dataset.xlsx.  
_problem1.M2_D4.csv_ is the result of NSGA2 iterating 10000 times with 800 population.It is used to calculate the GD and IGD of NSGA2-P.  
_moea_NSGA2_templet.py_ is a file in the path of \templates\MOEAs\NSGA2 in the python library geapy. We have revised it. The part commented out is the content before modification.    
_new\_pr.py_ is the start file used to run NSGA2-P.  
See the comments in the code for more details.  
