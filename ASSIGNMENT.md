Networks are dynamic and often we have access to incomplete data about their connections. As introduced in Lectures 7, link prediction is the fundamental problem of predicting missing or future connections between nodes in a network, based on the available data on network structure and nodes. Link prediction has several important applications across domains: in recommendation systems it is used to suggest products, movies, music, or content to users based on inferring missing links in user-item networks; in biology, link prediction can be used to predict interactions between proteins, aiding in drug discovery and understanding biological processes; in genetics it can predict potential genetic interactions or pathways; in transportation networks, link prediction can be used to forecast transportation demand, helping with urban planning and traffic management. In finance, link prediction can help detect fraudulent transactions; link prediction has also been used to identify missing links in criminal networks and help investigators. 

Link prediction constitutes a very convenient playground for you to test your data science and supervised learning skills, together with your knowledge on the ethical implications of data science. While the applications of link prediction extend to many domains, in this assignment we will focus on link prediction in social networks. In this case, nodes are people and links represent social connections such as friendship, family ties, professional collaborations, or common interests. One of the suggested readings for Week 4 is one of the first scientific papers addressing the problem of link prediction in social networks (linkDownload link). We suggest you check it before starting the assignment to have a better sense of the link prediction problem in social networks (and possible solutions).

Previous research in social sciences has revealed important regularities in social networks, which indicates that missing connections can possible be predicted: the clustering coefficient of social networks is high, meaning that if two individuals have a have a friend in common there is a high probability they are also connected; in social networks we observe homophily, meaning that connected individuals tend to be similar; finally, networks reveal well-defined communities (as discussed in lecture 7). Some of these principles might be relevant when defining features to predict missing social links (i.e., when doing feature engineering).
--

In Assignment 2 you will focus on two questions:

Can we predict missing links in a social network?

What are the ethical implications of link prediction?

Assignment 2 will be divided in three parts:

Part 1: Develop an algorithm for link prediction based on supervised learning (50%).

Part 2: Essay reflecting on the ethical aspects of your solution for link prediction (50%).

Part 3: Short reflection on the use of LLMs in the assignment (0%).
 

Part 1 and part 2 will be equality weighted in your final grade. Part 3 does not affect your grade (although it is mandatory section).

Part 1: A supervised learning approach to link-prediction (50%)
In the first part of this assignment you will develop a link-prediction algorithm based on the supervised learning methods you learned in this course. Link prediction can be framed as a classification problem, where an algorithm can be trained to predict the class of any possible link — existent (positive) or non-existent (negative). To tackle this problem you should follow some steps:

Feature engineering: First you should be able to import network data and define the features you believe are relevant to predict links in social networks. The data we will provide you consist of a social network where some links were removed. Each node will also be characterised by a categorical feature. Examples of important features might be the class that nodes belong to, or the number of common connections between nodes.
Defining a training, validation and test set: Once you have defined your feature space, you have to create your training, validation and test set. Your training set should be composed of positive examples (i.e., edges that exist in the network that was provided to you) and corresponding features. Your training set should also be composed of negative examples (i.e., edges that do not exist in the network provided). You will need to decide the ratio of positive and negative examples to include in your training set. You also have to decide about the split between training, validation and test set to train and select your best model. 
Model selection and validation: Once you have the required training and validation set, you have to select a model to be trained. You must use one of the methods discussed in the lectures (more complex models will not necessarily lead to higher performance, given the characteristics of the data we will provide you).
Application and test: Once you trained and validated your model, your solution should be able to decide if a (missing) link exists or not in a network. You will be welcome to submit your guesses to a Kaggle competition (details below). 
Note that we provide an auxiliary notebook (and auxiliary files) that can greatly help you in the previous steps. 

 

Data
The base dataset to use consists of a (synthetic) social network with 1800 nodes and 7085 links. This network was generated by removing ~10% of the links of an original network (only 6377 links are visible). Your task is to accurately guess whether a link given as an input belongs (or not) to the original network. All needed data can be found in the following zip fileDownload zip file.

Network data: Each node is numbered from 0 to N-1, where N is the number of nodes in the network. The network structure is provided as a list of edges (edges_train.edgelist). In this file, each line corresponds to an edge (or link): int1,int2 where int1 and int2 correspond to the identification of each node connected by the edge. The network is assumed to be undirected (i.e., if a connection between nodes 1 and 2 exists in the file, the connection 2 to 1 is also assumed to exist).
Node data: Each node is characterised by a categorical feature that can take 5 possible values ['b', 'a', 'c', 'e', 'd', 'f']. These might constitute a sensitive attribute (e.g., political affiliation, personal preferences, demographic data...). The feature of each node is provided in a .csv file (attributes.csv). In this file, each line corresponds to a node and the value in a given line corresponds to the feature of the corresponding node. 
Final test data: The ultimate goal of your algorithm is to determine if a link exists or not in the original network. In file solutionInput.csv you are provided a list of links. You should be able to output, for each link, a prediction: 1 if you believe the links exists in the original network; 0 if you believe the links does not exist in the original network. solutionInput.csv contains a total of 1416 links: 708 exist in the original network (positive examples); 708 do not exist in the original network (negative examples). Your ultimate goal is to develop an algorithm that achieves high accuracy in this test data. In Lab5 you will already produce an example of output that your (assignment 2) program should provide (example of Lab 5 output hereDownload here).
 

Kaggle competition
You are invited to participate in a Fundamentals of Data Science Kaggle competition with your solution. Please join the competition and form teams via this link: https://www.kaggle.com/t/0011f1e730534ecfb64631906c4941f5Links to an external site. 

Teams are formed via the "Team" tab. If you wish to participate, please create and enroll in the same Kaggle team with your team members.

Kaggle competitions are a fun way of testing and improving your data science skills. This assignment can be an opportunity for you to get familiar with these competitions and to have fun comparing your solution and your performance with alternative approaches by your colleagues. 

There is some history of link prediction competitions on Kaggle: in 2011 Facebook launched a recruitment Kaggle competition consisting on, precisely, a link prediction task: https://www.kaggle.com/c/FacebookRecruitingLinks to an external site. 

Participating in this competition is optional. To incentivise participation, we will give a bonus of 0.5 points to the 3 top-performing teams (as long as they achieve more than 0.8 as final accuracy score) and 0.1 to all participating teams that achieve a minimum of 0.8 in the final accuracy score. 

 

Part 2: Ethics essay (50%)
The second part of the assignment consists of a short essay related to ethics. The aim is to evaluate the model you created in Part 1 in terms of ethical implications. Take note that the model from Part 1 can be used in all kinds of setting, for example, to predict the likelihood of being a criminal or to offer social recommendations in social media. In these settings several ethical issues may arise. In the essay you need to elaborate on one selected ethical issue of using your model. You have to refer to ethical concerns related with your specific solution, and ways in which you could improve it. Such improvements should only be discussed in the essay and not coded.

 

Essay
The essay consists of maximum 5 pages, not including references (any page beyond the page limit won't be read and will lead to a penalty in the final grade). The essay must include 4 parts: Introduction, Analysis, Mitigation, Conclusion.

The Introduction should include a short description of potential problems of your model related to Fairness, Privacy and Transparency; 1 problem for each of these aspects. In addition, indicate which of these problems you investigate in more detail in the paper.

In the ‘Analysis’-part of the paper you are expected to analyze and explain the selected problem in more detail. For example, think about potential causes, explaining who is affected and potential impacts. If possible, connect to the topics introduced in the lectures. Also consider describing and connecting to similar cases.

In the ‘Mitigation’-part of the paper you are expected to describe and explain potential ways of mitigating and dealing with the problem. The lectures and similar cases could help here as inspiration. 

In the Conclusion you are expected to shortly summarize the paper and main reasoning. Also mention possible limitations in the paper. 

The paper has a 5 page limit excluding references. Suggestions for the different parts: Introduction is about ½ page. Analysis is about 2 pages. Mitigation is about 2 pages. Conclusion is about ½ page.

In the essay it is allowed to add your own opinions, but make clear when you are sharing your opinion (e.g. by using phrases like ‘Our viewpoint is …’ etc). Take note that in principle you need to describe the problem and mitigation of the problem as objectively as possible. Again: you need to focus on the details of your specific approach in Part 1.

Part 3: Reflection on LLM use (0%)
Finally, your report should finish with a reflection on the use of LLMs in your assignment. This section has one page max and include short answers to the following questions

Use description: How were LLMs used? (e.g., help coding? understand peer-code? ideation? proof-reading?). Please be transparent. 

Advantage(s) for the assignment: How do you believe the use of LLMs benefited your assignment? Could you have completed the assignment without Generative AI?

Advantage(s) for learning: How do you believe the use of LLMs benefited your learning process as a data scientist? 

Disadvantages: Were you dissatisfied by any type of LLM use form? 

(The results of this section will be used to guide future education directions and materials in this course. With this section we do not want to penalise AI use; note that this section has no impact in your grade.)


Report and Submission
Please submit a .zip file containing a .pdf report and the code (e.g., .ipynb notebook) you used in your link prediction task. Each group is expected to work independently in their code/notebook. Grading will be mostly based on the report and the code will only be checked in case any doubts arise on how the analysis was performed. The report should contain all the information for a reader to understand the analysis and results. Plagiarism detection will be used (the 'Regulations governing fraud and plagiarism for UvA students' applies to this course. This will be monitored carefully. Upon suspicion of fraud or plagiarism, the Examinations Board of the programme will be informed).

Your report should have maximum 11 pages. It should describe your approach to link prediction (max 5 pages), your ethics essay (max 5 pages), and reflection on LLM/GenAI use (max 1 page). List of references do not count towards this limit. 

Please note that any page beyond the page limit won't be read and will lead to a penalty in the final grade. The report styles are loosely based on the style of computer science scientific journal papers. The basic guideline for the lay out of your report is as follows:
 
Title

Author List (Mandatory, alphabetic order)

Abstract: Summary of your approach to link prediction and ethical issue discussed

Technical solution (max 5 pages): Describe your supervised learning solution to the problem of link prediction. You should have the following short sub-sections

Feature engineering: which features were selected and why
Data preparation: how did you go about creating a training/validation/test set and splitting data 
Model selection: which supervised learning model was selected and why
Model training and validation: describe how you selected hyperparamers and validated your model
Final results: report the final accuracy metrics in your test set
Limitations: reflect on technical limitations of your approach   
Ethics essay (max 5 pages): Reflect on ethical issues related with link prediction and with your solution in particular. As described above you should have the following sub-section:

Introduction
Analysis
Mitigation
Conclusion
 

Reflection on LLM use (max 1 page): reflection on the use of LLMs in your assignment:

Use description
Advantage(s) for the assignment
Advantage(s) for learning
Disadvantages
You can find the report template in LaTeXDownload LaTeX and WordDownload Word. If you are having troubles compiling the LaTeX files in your computer, feel free to use overleaf. Do not deviate from the template (i.e., do not modify font size, spacing, number of columns).