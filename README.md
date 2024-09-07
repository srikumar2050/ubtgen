# UBTGen: Utility-based transaction data generator

The repository provides the source code of utility-based transaction data generator. This module is used as part of the classifier modeling method (HUG-IML) for transaction generation. High Utility Gain-Interpretable Machine Learning (HUG-IML) is an intrinsic classifier model that extracts a class of higher order patterns and embeds them into an interpretable learning model such as logistic regression. The model supports both binary and multi-class classification problems. The specific details of the HUG-IML models, benchmark results, and their applications can be referred to in the IEEE Access paper titled: Interpretable classifier models for decision support using high utility gain patterns, IEEE Access 2024, DOI: https://doi.org/10.1109/ACCESS.2024.3455563.

If you use the software programs in this repository, please cite the following paper:

```
    @article{krishnamoorthy2024,
                title={Interpretable classifier models for decision support using high utility gain patterns},
                author={Krishnamoorthy, Srikumar},
                journal={IEEE Access},
                year={2024},
                doi={https://doi.org/10.1109/ACCESS.2024.3455563}
    }
```

<br/>

### 1. REPOSITORY INFORMATION

This repository primarily contains java files for generating utility-based transaction data from any supervised learning dataset. The specific details of program/data files and directories contained in this repository are as follows:

#### 1.1 Benchmark datasets

supervisedLearningDatasets: This directory contains four benchmark binary classifier modeling datasets:

1. Portugese Bank Telemarketing: UCI (https://archive.ics.uci.edu/dataset/222/bank+marketing)
2. Home Equity Line Of Credit (HELOC): FICO (https://community.fico.com/s/explainable-machine-learning-challenge)
3. Pima Indian Diabetes: National Institute of Diabetes and Digestive and Kidney Diseases (http://archive.ics.uci.edu/ml, https://data.world/uci/pima-indians-diabetes)
4. Titanic: Open ML (https://www.openml.org/search?type=data&sort=runs&id=40945&status=active)

#### 1.2 Java related files

Binary files

UBTGen.jar: Compiled java files. Java(TM) SE Runtime Environment (build 22.0.1+8-16) was used for compiling the java files.

Source files

UBTGen_src: This directory contains the source code of the java files used for transaction generation. Use these files if you wish to manually compile the program on your JVM or customize it as per your needs. The specific source files included are:

    * RunUBTGen.java                          The class that is invoked by the UBTGen.jar file
    * UtilityBasedTransactionGenerator.java   The core class that generates transactions
    * MinMaxScaler.java                       Perform min-max scaling transformations
    * KBinsDiscretizer.java                   Discretize the numerical variables based on user-specified or estimated bins
    * LabelBinarizer.java                     Program to encode categorical data
    * NMI.java                                Program to compute Normalized Mutual Information
    * CorrelationCustom.java                  Pearson correlation computations
    * META-INF/MANIFEST.MF                    Specifies the main class used by the jar file

#### 1.3 Configuration files

config: This directory includes the configuration files used for transaction generation. The configuration follows a key=value format. The configuration files are read by the RunUBTGen.java program. The details of keys used in the config file are as follows:

    * inputFile                 input supervised learning dataset
    * outputFile                location where the output transactions should be written
    * header                    whether the file has a header row (true or false)
    * delimiter                 delimiter to be used for reading data (default: ,)
    * targetColIndex            index of the coulmn that contains the supervised label or target data
    * skipColsIndices           list of columns that should be skipped while generating transactions
    * numericIntColsIndices     list of columns that hold integer-valued data
    * numericFloatCols          list of columns that hold real-valued data
    * catColsIndices            list of columns that hold categorical or nominal-valued data
    * B                         number of bins to be used for discretization of numeric columns
    * writeTransformParameters  whether the data transformation parameters learnt during scaling, discretization, and encoding needs to be stored in a separate file
    * missingValueImputation    whether missing value imputation should be performed - missing values are replaced with mean (mode) for numeric (categorical) valued data.

A sample config file is provided below for the pima indian diabetes dataset:

      inputFile=./supervisedLearningDatasets/pima indians diabetes.csv
      outputFile=./outputs/pima_utility.csv
      header=true
      delimiter=,
      targetColIndex=8
      skipColsIndices=
      numericIntColsIndices=0,1,2,3,4,7
      numericFloatColsIndices=5,6
      catColsIndices=
      B=5
      writeTransformParameters=false
      missingValueImputation=false

The missing value imputation is an additional functionality included in this stand-alone utility-based transaction data generator. The current implemenetation of the HUG-IML classifier do not support missing values. 

#### 1.4 Data related files

outputs directory: This directory stores the generated transaction files. It also stores the generated parameter files, if the writeTransformParameters option is set to true in the configuration file.

#### 1.5 License information

GNU GPLv3 License: This repository contains a free software program. You can redistribute it and/or modify it under the terms of GNU General Public Licence. The license details are shared in this file. It can also be referred to online at http://www.gnu.org/licenses/.

<br/>

### 2. USAGE GUIDELINES

1. Prepare (or) use the shared jar file. 
UBTGen.jar shared in the repository is prepared by compiling java files using Java(TM) SE Runtime Environment (build 22.0.1+8-16). If your JDK/JRE is incompatible with this version, then you may have to compile the java files (refer to UBTGen_src directory for the java files). Compile the java program and create jar file using the following steps:
   a. javac *.java
   b. jar cvfm UBTGen.jar META-INF/MANIFEST.MF *.class

2. Run the jar file to generate transactions with utility information.

   ```
   #use the default configuration file, config/configTitanic.txt 
   java -jar UBTGen.jar`
   ```
   
   On successful run, titanic_utility.txt file will get generated in the outputs folder.

   ```
   #specify the configuration file     
   java -jar UBTGen.jar config/configPima.txt
   ```

   On successful run, pima_utility.txt file will get generated in the outputs folder.

    ```
    The format of the output file follows a standard High Utility Itemset (HUI) dataset format i.e. each line of the transaction has the form: 
    {list of items}:total transaction utility:{list of utilities of items} 
    e.g. 1 3 4 5:2.52:1.00 1.20 0.12 0.20
    ```

For inspecting the intermediate parameters learnt, set the writeTransformParameters=true in the config file and re-run the program.
