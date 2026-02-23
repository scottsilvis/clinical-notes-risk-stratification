- [Rationale](#rationale)
- [Data Explained](#data-explained)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [NLP (Natural Language Processing)](#nlp-natural-language-processing)
  - [Overview](#overview)
  - [TF-IDF (Term Frequency-Inverse Document Frequency)](#tf-idf-term-frequency-inverse-document-frequency)
  - [Combined Model](#combined-model)
- [Python Scripts](#python-scripts)
  - [data.py](#datapy)
  - [model.py](#modelpy)
    - [load\_joined\_clinical\_data](#load_joined_clinical_data)
    - [make\_train\_test\_split](#make_train_test_split)
    - [run\_baseline](#run_baseline)
    - [run\_text\_baseline](#run_text_baseline)
    - [run\_combined\_model](#run_combined_model)
    - [main](#main)



# Rationale 

While I have years of experience with various statistical testing and exploratory data analysis, I 
have had limited opportunity to work directly with language models in prior roles. To build 
experience in this area, I decided to use a combination of self teaching and AI to help me build a project to expose me to these areas of Data Science that I am less famaliar with. 

In this way, I built a prototype to predict 30-day readmission risk using structured patient 
variables, then extended it with NLP features from clinical notes to quantify lift. The data is 
fully synthetic, and I created three .csv files to represent three sources of data that I might see 
in a real world application. These three files are patients.csv, outcomes.csv, and notes.csv.

---

# Data Explained

The patients.csv dataset contains many variables that one might know before discharge. eg. age, 
comorbidity burden, length of stay, prior admissions. 

outcomes.csv contains the simulated readmission outcome (readmit_30). 

notes.csv contains the Natural Language Processing data to be added to the model after baseline. 

---

# Exploratory Data Analysis

In the EDA phase, I aim to answer 5 questions:

1. What percentage of patients are readmitted?
   - If the rate of readmitance is too low it could change the types of analyses we need to do
2. Does the data make sense conceptually? 
   - For example, increase comorbidity theoreticall should lead to increase readmittance 
3. Are there any obvious data issues?
   - missing data, weird ranges, etc
4. How seperable are the outcome categories?
   - if we have distinct seperate peaks it bodes well for the model performing well
  
Refer to 01_eda.ipynb for the code used to answer the above questoins. 
  
During EDA I saw a moderately balanced readmission outcome with reasonable feature distributions 
and no major data quality issues. Individual structured predictors demonstrate limited univariate 
separability, although baseline modeling suggests meaningful signal when features are combined. 
Because structured features alone showed limited univariate separability, this analysis motivated the inclusion of unstructured text as a potential source of additional signal.

---

# NLP (Natural Language Processing)

## Overview

The entire point of this section is to answer a single questoin. Do the clinical notes differ by 
outcome in a way that adds predictive signal beyond structured data? I started with a structured 
baseline (model_a = AUC 0.670). Two things need to happen for this to be a successful addition. The 
first is I must create a text-only model (model_b) to validate that clinical notes encode 
outcome-relevant signal at all. The dataset will be considered outcome-relevant if the model_b 
AUC > 0.50. This is the threshold that signifies random behavior of the model. Even if the 
text-only model underperforms the structured baseline, the key test is whether combining the text 
and structured models (model_c) produces an AUC lift (meaning AUC model_c > AUC model_a). 

## TF-IDF (Term Frequency-Inverse Document Frequency)

Because clinical notes are unstructured text, they cannot be used directly by a statistical 
classifier. The first step in NLP modeling is therefore feature extraction: converting free-text 
notes into numerical features that retain clinically meaningful information. For this baseline 
analysis, I use Term Frequency–Inverse Document Frequency (TF-IDF), a standard and interpretable 
method for representing text.

TF-IDF represents text by weighting words based on how frequent they are in a document but how rare 
they are across the corpus, which helps emphasize informative terms while down-weighting 
boilerplate language.

A text-only TF-IDF + logistic regression baseline achieved ROC-AUC 0.679, demonstrating that 
clinical note language encodes outcome-relevant information. This is slightly higher than the 
structured-only baseline (0.670), motivating a combined model to test incremental value beyond 
structured data.

## Combined Model

The final step is to build the combined model, but before I can do that, I need to rewrite portions 
of model.py in an attempt to make it more usablein the future, but also to add some sanity checks. 
Because the split dataset section of the function is duplicated in both the baseline models, and by 
extension would be necessary in the combined model, I decided to pull that out and make it its own 
function. This should have the benifit of ensuring that all three models are working on exactly the 
same data. I converted the sex variable from categorical (M/F) to binary(1/0), otherwise it broke 
the combind model. Because this transformation preserves category meaning rather than redefining 
it, I chose to apply it upstream so that all models operate on a consistent dataset.

All three models now work. I started with a structured baseline (AUC 0.670), then built a text-only 
TF-IDF model (AUC 0.679) to confirm that notes contained outcome-relevant signal. Finally, I 
combined structured and text features using a unified pipeline with a shared train/test split, 
which produced an AUC of 0.687. This demonstrates that clinical notes add incremental predictive 
value beyond structured data.

# Python Scripts

## data.py

The data.py file is responsible for creating the three datasets used in this project. While the 
initial version of this file was generated with the help of an AI model, it was reviewed 
line-by-line closely to ensure I understood how the data were being created, as this is important 
for justifying the modeling decisions later in the project.

The script accepts three command-line inputs: --generate, --n_patients, and --seed. The --generate 
flag tells the script to actually create the data. If no values are supplied for the other 
arguments, the script defaults to generating a dataset with 1,500 patients using a fixed random 
seed of 7. Using a fixed seed ensures the dataset is reproducible, meaning the same data will be 
generated every time the script is run with the same inputs. Changing the seed produces a different 
dataset with the same overall statistical structure.

One section of the file is dedicated to generating clinical notes for downstream NLP modeling. 
This section defines several groups of phrases: base phrases, low-risk phrases, high-risk phrases, 
and filler phrases. Each clinical note begins with a small set of base phrases that mimic templated 
discharge documentation. The body of the note is then constructed by repeatedly selecting phrases 
from either the low-risk or high-risk lists. The choice of which type of phrase to include is 
driven by the patient’s underlying risk: for each phrase, a random number is compared to the 
patient’s risk value, and higher-risk patients are more likely to receive high-risk phrases. This 
process is repeated a random number of times (between 4 and 9) to introduce variability in note 
length. In some cases, an additional filler sentence is appended to further diversify the notes. 
The final note is created by joining the base phrases and selected content into a single text 
string.

Another section of the file generates the structured patient variables. Patient age is drawn from 
a normal distribution and constrained to realistic values. Sex is assigned randomly with a slight 
imbalance (52% female, 48% male). Comorbidity count and prior admission count are generated using 
Poisson distributions, which are appropriate for modeling count-based variables. Length of stay is 
generated using a gamma distribution, which produces a right-skewed distribution that better 
reflects real hospital stays, where short stays are common and long stays are less frequent.

A latent readmission risk score is then calculated as a weighted combination of these structured 
variables. The weights are chosen to reflect plausible clinical relationships, such as higher risk 
with increasing comorbidity burden, more prior admissions, longer length of stay, and older age. 
Random noise is added to this score to prevent the data from being unrealistically clean and to 
represent unmeasured clinical and social factors. This risk score is converted into a probability 
using a logistic transformation, ensuring the result falls between 0 and 1. Finally, the binary 
readmission outcome (readmit_30d) is generated by randomly sampling based on this probability, so 
patients with higher risk are more likely—but not guaranteed—to be readmitted.



## model.py

### load_joined_clinical_data

The load_joined_clinical_data function requires a path variable, which is used to read the 
patients.csv, notes.csv, and outcomes.csv files. These three datasets are merged using joins on patient_id. This  ensures that each row of patient data is correctly aligned with its corresponding 
note and outcome, rather than relying on row order alone. Sanity checks occure before nad after the join, and the final dataframe is returned. 

### make_train_test_split

The function make_train_test_split takes a dataframe and splits it into a training and testing set. 
The function accepts the following parameters: df (the input dataframe), target_col (the name of 
the target column to stratify on) which defaults to "readmit_30d", test_size (the percentage of the 
data to include in the test set) which defaults to 0.25, and seed (the random state for 
reproducibility) which defaults to 7. The function uses the train_test_split function from 
scikit-learn to perform the split, stratifying on the target column to preserve class distribution. 
The function returns the training and testing dataframes.

### run_baseline

The run_baseline function requires a path variable, along with a random seed that defaults to 
seven. The primary purpose of this function is to train and evaluate a baseline model. It begins by 
calling load_joined_clinical_data to retrieve the dataframe. The data are then split into training 
and testing sets using the make_train_test_split helper function. A logistic regression model is 
instantiated with a maximum of 1,000 optimization iterations and then fit to the training data. 
Model performance is evaluated by generating predicted probabilities for the test set and computing 
the ROC-AUC by comparing those probabilities to the true test labels. This value is printed to the 
terminal, and an ROC curve is generated and saved as a figure.

### run_text_baseline

The function run_text_baseline takes the variable processed_dir and uses it when it calls the 
function load_joined_clinical_data. The function run_text_baseline also accepts the variable 
seed, which it uses when calling the function make_train_test_split. run_text_baseline accepts 
the datafame returned by load_joined_clinical_data and immedately passes it to 
make_train_test_split. The output of this is train_df and test_df. This is this split into 
z_train, y_train, z_test, and y_test. The package scikit-learn is used to create a pipeline that 
consists of two steps: a TfidfVectorizer and a LogisticRegression. The vectorizer is configured 
to consider both unigrams and bigrams, convert all text to lowercase, and removes English stop 
words. TF-IDF assigns each word or phrase a unique index and represents each document as a sparse 
vector of TF-IDF weights rather than raw counts. These vectors are then used as inputs to the 
classifier. A logistic regression model is then fit to this data along with the outcome data. The 
model is then used to predict probabilities on the test set, and the ROC-AUC score is calculated 
and printed. A ROC curve is then plotted using the true labels and predicted probabilities. The 
figure is saved to the out_dir directory as roc_text_baseline.png.

### run_combined_model

The function run_combined_model takes the variable processed_dir and uses it when it calls the 
function load_joined_clinical_data. This function also accepts the variable seed, which it uses 
when calling the function make_train_test_split. run_combined_model accepts the datafame 
returned by load_joined_clinical_data and immedately passes it to make_train_test_split. The 
output of this is train_df and test_df. This is this split into numeric and text features, which
are used to make the X_train and X_test variables. The preprocessor is defined as a 
ColumnTransformer that applies a TfidfVectorizer to the text feature (note_text) and passes 
through the numeric features. The TfidfVectorizer is configured to consider both unigrams and 
bigrams, convert all text to lowercase, and removes English stop words. The package scikit-learn 
is used to create a pipeline that the above-mentioned preprocessing steps and a Logistic 
Regression. The model is then used to predict probabilities on the test set, and the ROC-AUC 
score is calculated and printed. A ROC curve is then plotted using the true labels and predicted 
probabilities. The figure is saved to the out_dir directory as roc_combined.png.

### main

The final function, main, serves as the entry point for the script. It defines the directory 
structure used by the model and triggers execution of the modeling workflows when the 
appropriate command-line flag is provided. 

As the project evolved, this script was extended to support not only baseline, but text-only and 
combined models. To ensure fair comparison, I refactored the train/test split logic into a shared 
helper function (as seen above) so that all models operate on the exact same data partition. This 
avoids split-induced performance differences and allows changes in ROC-AUC to be attributed to 
features rather than sampling noise.

The next step of the project is to perform Exploratory Data Analysis to confirm the model before we 
add the complexity of the language models. 

---





