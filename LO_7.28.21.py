
# coding: utf-8

# # Imports

# In[2]:

## Loading Packages and Libraries

# For unzipping data package
import _pickle as cPickle
import gzip
from collections import defaultdict
import random

# For analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# RDKit imports

from rdkit import Chem #This gives us most of RDkits's functionality
from rdkit.Chem.Draw import IPythonConsole #This will help us interface with jupyter
from rdkit.Chem import Descriptors, PandasTools, AllChem #Let's us describe molecules (e.g. molecular weight), and interface with pandas
IPythonConsole.ipython_useSVG=True  #SVG's tend to look nicer than the png counterparts

# imports for RF algorithm
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier


# In[3]:

# Load class names and labels

dataDir = "/class/datamine/corporate/cistar/Shared/RxnData/data/"

with open(dataDir+"reactionTypes_training_test_set_patent_data.pkl",'rb') as f:
    reaction_types = cPickle.load(f)
with open(dataDir+"names_rTypes_classes_superclasses_training_test_set_patent_data.pkl",'rb') as f:
    names_rTypes = cPickle.load(f)

    
# Load 40000 rxn training data set

infile = gzip.open(dataDir+'training_set_patent_data.pkl.gz', 'rb')

fps_train=[]

lineNo=0
while 1:
    lineNo+=1
    try:
        smi,lbl,klass = cPickle.load(infile) 
        if not lineNo%800:
            print(klass, lineNo)
    except EOFError:
        break
    fps_train.append([smi,lbl,klass])
#     if not lineNo%10000:
#         print("Done "+str(lineNo))        
#     print(smi[1])


# In[4]:

# Load 10000 rxn test data set

infile = gzip.open(dataDir+'test_set_patent_data.pkl.gz', 'rb')

fps_test=[]

lineNo=0
while 1:
    lineNo+=1
    try:
        smi,lbl,klass = cPickle.load(infile) 
        if not lineNo%200:
            print(klass, lineNo)
    except EOFError:
        break
    fps_test.append([smi,lbl,klass])
#     if not lineNo%10000:
#         print("Done "+str(lineNo))        
#     print(smi[1])


# In[5]:

infile = gzip.open(dataDir+'unclassified_patent_data.pkl.gz', 'rb')
fps_unclass=[]

lineNo=0
while 1:
    lineNo+=1
    try:
        smi,lbl,klass = cPickle.load(infile) 
    except EOFError:
        break
    fps_unclass.append([smi,lbl,klass])


# In[6]:

# Make data into a pandas array

column_names = ['Smiles','Patent Number','Rxn Class']
training_df = pd.DataFrame (fps_train, columns=column_names)
test_df = pd.DataFrame (fps_test, columns=column_names)
unclass_df = pd.DataFrame (fps_unclass, columns = column_names)

# Convert Smiles strings to reaction objects

from rdkit.Chem import rdChemReactions
training_df['Reaction'] = training_df['Smiles'].apply(rdChemReactions.ReactionFromSmarts)
test_df['Reaction'] = test_df['Smiles'].apply(rdChemReactions.ReactionFromSmarts)
unclass_df['Reaction'] = unclass_df['Smiles'].apply(rdChemReactions.ReactionFromSmarts)


# 
# 
# # Defining functions using RDKit

# In[7]:

# Create dictionary of all Molecular Fingerprinting types with names
fptype_dict = {"AtomPairFP": AllChem.FingerprintType(AllChem.FingerprintType.AtomPairFP),
               "MorganFP": AllChem.FingerprintType(AllChem.FingerprintType.MorganFP), 
               "TopologicalFP": AllChem.FingerprintType(AllChem.FingerprintType.TopologicalTorsion), 
               "PatternFP": AllChem.FingerprintType(AllChem.FingerprintType.PatternFP), 
               "RDKitFP": AllChem.FingerprintType(AllChem.FingerprintType.RDKitFP)}

# Functions for generating fingerprints
from rdkit.Chem import rdFingerprintGenerator

# Include agents in the fingerprint as either a reactant or product
## Inputs are reaction object, fp_type object, int, int
def diff_fpgen_withagents(rxn,fp_type,agent_weight,nonagent_weight):
    params = AllChem.ReactionFingerprintParams()
    params.fptype = fp_type
    params.includeAgents = True
    params.agentWeight = agent_weight
    params.nonAgentWeight = nonagent_weight
    fp = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn,params)
    return fp

# No agents included in fingerprint
## Inputs are reaction object, fp_type object
def diff_fpgen(rxn,fp_type):
    params = AllChem.ReactionFingerprintParams()
    params.fptype = fp_type
    params.includeAgents = False
    fp = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn,params)
    return fp

# Agent feature FP as taken fromSchneider additional materials
def create_agent_feature_FP(rxn):    
    rxn.RemoveUnmappedReactantTemplates()
    agent_feature_Fp = [0.0]*9
    for nra in range(rxn.GetNumAgentTemplates()):
        mol = rxn.GetAgentTemplate(nra)
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSSSR(mol)
        try:
            ri = mol.GetRingInfo()
            agent_feature_Fp[0] += Descriptors.MolWt(mol) #
            agent_feature_Fp[1] += mol.GetNumAtoms() #
            agent_feature_Fp[2] += ri.NumRings() #
            agent_feature_Fp[3] += Descriptors.MolLogP(mol) #partition coefficient
            agent_feature_Fp[4] += Descriptors.NumRadicalElectrons(mol) 
            agent_feature_Fp[5] += Descriptors.TPSA(mol)#topological polar surface area
            agent_feature_Fp[6] += Descriptors.NumHeteroatoms(mol)
            agent_feature_Fp[7] += Descriptors.NumHAcceptors(mol)
            agent_feature_Fp[8] += Descriptors.NumHDonors(mol)
        except:
            print ("Cannot build agent Fp\n")
    return agent_feature_Fp

# Define function for converting fingerprint vectors into arrays
from rdkit import DataStructs
def fingerprint2Numpy(finger):
    # input rdkit fingerprint (type UIntSparseIntVect) and output numpy array
    # dummy array to fill
    fp_np = np.zeros((1,))
    # Convert fingerprint into array to fill the dummy
    DataStructs.ConvertToNumpyArray(finger,fp_np)
    # Returns the filled array
    return fp_np


# # Defining the RF Algorithm

# In[8]:

model_final = RandomForestClassifier(max_depth=30,n_estimators=140,random_state=0)


# In[9]:


def accuracy(X,y,sp,r):
    # Cross validation
    cv = RepeatedStratifiedKFold(n_splits=sp, n_repeats=r)
    n_scores = cross_val_score(model_final, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    
    
def test_training_performance(report):
    
    fscore = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    
    # report performance
    print('Fscore: %.3f , Precision: %.3f , Recall: %.3f' % (fscore, precision, recall))
    
def performance(X,y):
    # Split sample data into training and testing subsets    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = model_final
    model.fit(X_train,y_train)
    y_predict = model.predict(X_test)

    # Using built in report 
    from sklearn.metrics import classification_report
    report = classification_report(y_test,y_predict,output_dict=True)
    
    fscore = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    
    # report performance
    print('Fscore: %.3f , Precision: %.3f , Recall: %.3f' % (fscore, precision, recall))
    
def fscore_classes(report):
    
    # Headers
    print(f"{'Name:':<40} F1-Score:")
    print("")

    # Print name of rxn and its f1 score for each rxn in the dict
    for i in report :
        try : 
            rxn_name = names_rTypes[i]
            fscore = report[i]['f1-score']
            print(f"{ rxn_name :<40} {fscore:.3f}")
        except :
            # pass on any items that do not show up in names_rTypes
            pass
#
        


# In[1]:

get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

def labelled_cmat(cmat,labels,figsize=(20,15),labelExtras=None, dpi=600,threshold=0.01, xlabel=True, ylabel=True, rotation=90):
    
    rowCounts = np.array(sum(cmat,1),dtype=float)
    cmat_percent = cmat/rowCounts[:,None]
    #zero all elements that are less than 1% of the row contents
    ncm = np.log(cmat_percent*(cmat_percent>threshold))

    fig = plt.figure(1,figsize=figsize,dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    fig.set_size_inches(figsize)
    fig.set_dpi(dpi)
    pax=ax.pcolor(ncm,cmap="YlGnBu")
    ax.set_frame_on(True)
  
    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(cmat.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(cmat.shape[1])+0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    if labelExtras is not None:
        labels = [' %s %s'%(x,labelExtras[x].strip()) for x in labels]
    
    ax.set_xticklabels([], minor=False) 
    ax.set_yticklabels([], minor=False)

    ax.set(ylabel="True Label", xlabel="Predicted Label")
    ax.xaxis.label.set_fontsize(10)
    
    if xlabel:
        ax.set_xticklabels(labels, minor=False, rotation=rotation, horizontalalignment='left',fontsize=10) 
    if ylabel:
        ax.set_yticklabels(labels, minor=False,fontsize=12)

    ax.grid(True)
    fig.colorbar(pax)
    plt.axis('tight')


# # CMAT

# In[13]:

# Defining input variables 

training_df['Diff FP'] = training_df['Reaction'].apply(diff_fpgen, fp_type=fptype_dict['MorganFP'])
rxns_fp_train = np.array([fingerprint2Numpy(finger) for finger in training_df["Diff FP"]])
agents_fp_train = np.array([create_agent_feature_FP(x) for x in training_df['Reaction']])
fingerprints_train = np.concatenate((rxns_fp_train, agents_fp_train), axis=1)

X_train = fingerprints_train
y_train = np.ravel(training_df[['Rxn Class']])

test_df['Diff FP'] = test_df['Reaction'].apply(diff_fpgen, fp_type=fptype_dict['MorganFP'])
rxns_fp_test = np.array([fingerprint2Numpy(finger) for finger in test_df["Diff FP"]])
agents_fp_test = np.array([create_agent_feature_FP(x) for x in test_df['Reaction']])
fingerprints_test = np.concatenate((rxns_fp_test, agents_fp_test), axis=1)

X_test = fingerprints_test
y_test = np.ravel(test_df[['Rxn Class']])

model_real = RandomForestClassifier(max_depth=30,n_estimators=140,random_state=0)
model_real.fit(X_train,y_train)
y_predict_real = model_real.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
my_report_real = classification_report(y_test, y_predict_real,output_dict=True)
my_cmat_real = confusion_matrix(y_test,y_predict_real)


# In[2]:

labelled_cmat(my_cmat_real,model_real.classes_)


# # Predictions for Unlabeled Data

# In[10]:

# Defining input variables 

training_df['Diff FP'] = training_df['Reaction'].apply(diff_fpgen, fp_type=fptype_dict['MorganFP'])
rxns_fp_train = np.array([fingerprint2Numpy(finger) for finger in training_df["Diff FP"]])
agents_fp_train = np.array([create_agent_feature_FP(x) for x in training_df['Reaction']])
fingerprints_train = np.concatenate((rxns_fp_train, agents_fp_train), axis=1)

X_train = fingerprints_train
y_train = np.ravel(training_df[['Rxn Class']])

unclass_df['Diff FP'] = unclass_df['Reaction'].apply(diff_fpgen, fp_type=fptype_dict['MorganFP'])
rxns_fp_unclass = np.array([fingerprint2Numpy(finger) for finger in unclass_df["Diff FP"]])
agents_fp_unclass = np.array([create_agent_feature_FP(x) for x in unclass_df['Reaction']])
fingerprints_unclass = np.concatenate((rxns_fp_unclass, agents_fp_unclass), axis=1)

X_unclass = fingerprints_unclass

model_real = RandomForestClassifier(max_depth=30,n_estimators=140,random_state=0)
model_real.fit(X_train,y_train)
y_predict_unlabeled = model_real.predict(X_unclass)
y_predict_proba = model_real.predict_proba(X_unclass)


# In[11]:

get_ipython().magic('matplotlib notebook')
predicted_labels = pd.DataFrame(y_predict_unlabeled, columns = ['PredClass'])
prediction_proba = pd.DataFrame(y_predict_proba, columns = model_real.classes_)


# In[12]:

labeled_set = pd.concat([unclass_df,predicted_labels,prediction_proba],axis=1)


# In[14]:

histo = []
for i in range(len(labeled_set)):
    prediction = str(labeled_set.loc[i].at['PredClass'])
    prob = labeled_set.loc[i].at[prediction]
    histo.append(prob)


# In[26]:

patents = labeled_set[['Patent Number']]
labels = labeled_set[['PredClass']]
probs = pd.DataFrame(histo,columns=['Probability'])

trevor = pd.concat([patents, labels, probs],axis=1)
trevor.to_csv('Lainey_PredictedLabels.csv')


# In[14]:

get_ipython().magic('matplotlib notebook')
plt.hist(histo)
plt.style.use('seaborn')
plt.xlabel('Prediction Probability')
plt.ylabel('Count')
plt.savefig('unlabeled_histo.png',dpi=300)


# In[15]:

# Let's filter the predictions to only keep those over 90% probability
filtered_predictions = prediction_proba > 0.90
filtered_predictions = filtered_predictions[~((~filtered_predictions).all(axis=1))]

labeled_set_filtered = pd.concat([unclass_df,predicted_labels,filtered_predictions],axis=1).dropna()
print(len(labeled_set_filtered))
labeled_set_filtered.head()


# In[16]:

get_ipython().magic('matplotlib notebook')
labeled_set_filtered.PredClass.value_counts(sort=False).plot(kind='bar',figsize=(20,12),fontsize=12)


# # Sample reactions

# In[22]:

labeled_set_filtered[labeled_set_filtered.PredClass=='2.2.3']


# In[23]:

def rxn_check(row):
    prediction = str(labeled_set.loc[row].at['PredClass'])
    print(labeled_set.loc[row].at['Patent Number'])
    print(prediction)
    print(labeled_set.loc[row].at[prediction])
    rxn = labeled_set.loc[row].at['Reaction']
    return(rxn)


# In[29]:

rxn_check(45626)


# In[81]:

rxn_check(896)


# In[82]:

rxn_check(1665)


# In[83]:

rxn_check(5203)


# In[84]:

rxn_check(2253)


# In[89]:

rxn_check(25291)


# In[91]:

rxn_check(24404)

