# AIAP Technical Assessment — Decision Log

**Candidate name (as in NRIC): Ong Jian He**

**Email (as used in your application): faithwish@hotmail.com**

\---

## A note on this document

This decision log is the primary instrument by which we understand your thinking. The questions below cover the reasoning behind your work — from how you define the problem at the start to the decisions you made during the work itself. Do answer all five questions in your own words.

You may use AI assistance freely on the technical deliverables (the EDA and the ML pipeline), but this Decision Log itself should be written by you — it is the record of your own thinking that we cross-check against your chat history.



\---

## 1\. Clarifying questions

What questions would you ask to better define and narrow the problem statement? For each question, briefly explain how the answer would meaningfully change your approach.

Note: If it helps your decision-making, you may assume and list out the stakeholders' likely answers.



**Your answer:**



1. Google \& Agoda review by customer for Travelers’ Inn competitor.
This will help us understand why Travelers’ Inn competitor where they have done well to earn their good rating
2. Monthly room rates and occupancy rates for each room rate for Travelers’ Inn and Travelers’ Inn competitor
This will help us understand whether Travelers’ Inn is pricing the correct pricing for each room and season period
3. Staff ID and customer ID

&#x20;   This will help us understand staff and customer behaviour which will in understand Travelers’ Inn staff service level and customer behaviour







\---

## 2\. Defining the Problem Statement

Restate, in your own words, the refined problem you decided to solve. List your key assumptions. Briefly note what other framings you considered, and what you deliberately left out or scoped down, and why.



**Your answer:**



Occupancy rate in the hotel remain low despite having mixed feedback from customer and there is high number of booking being called off which lead to lost of revenue each month.





\---

## 3\. Key decisions during Solution Development

Walk through three key decisions you made during Solution Development. For each: what options did you consider, what did you choose, and why? These can be technical (modelling choices, feature handling, evaluation metrics) or about the work itself (what to prioritise, what to drop, how to spend your time).



**Your answer:**





**Data analysis for missing data \& General automation**



General automation use where it will load the file based on db file type instead of specific file name so that the code can be reuse. Remove duplicates in the dataset, removes: leading spaces \& trailing spaces but keeps spaces inside the text, Convert all text to lowercase, Apply absolute value only to numeric columns.



Check if the missing data is missing at random using CHI-SQUARE TEST and percentage of missing data. Dropping throws away incomplete records to preserve data integrity, model and meadian median is another way to fill in missing data while IterativeImputer is a multivariate machine learning technique that predicts and fills in missing values based on relationships with other features. Therefore the assumption in this task is use IterativeImputer as data is Missing at Random (MAR), IterativeImputer is almost always better than removing missing data. Removing data introduces statistical bias and reduces your sample size, whereas IterativeImputer uses relationships in your other features to estimate the true values accurately. Dropping nearly 70% of your dataset will severely damage your machine learning model by destroying statistical power and introducing massive bias.



For this task, rating is crucial which is based on customer feedback and using IterativeImputer to calculate rating may lead to unreliable result. Therefore before IterativeImputer is used, machine learning is used to train and learn on the comment data to provide predicted rating. This is only for data where there is comment but no rating or does not have bother comment \& rating.



Therefore in this Solution Development, I have decided to use machine learning to predict the rating based on comment and after that use IterativeImputer to impute the missing data as most of the data are missing at random as well as TfidfVectorizer \& CountVectorizer to have AI summary of the comment.



**Feature engineering**

1. Correct data error in the dataset
2. Convert to binary for data set that only has word data
3. Predict rating for empty rating column data from comment column
4. ITERATIVE IMPUTER to fill in the missing data
5. Rule-based grouping for specific column (arrival\_date\_month using season, country group by contingent, adr \& previous\_cancellations group by range based on low-mid-high, days\_in\_waiting\_list group short, medium, long)
6. Advance feature engineering - Create Family\_size using adult, child \& baby column
7. Advance feature engineering - Create client behaviour through comment rating and top 10 client with bad rating
8. Advance feature engineering - Group comment by clustering which help to understand the type of comment 





**Modelling evaluation use**

3 models are use using the base features engineering column data to predict the probability of service level rating and recommend which machine learning model to use based on F1 and roc\_auc score.



The model with the best F1 Score and ROC AUC will be recommended to use.



Note: F1 Score and ROC AUC (Area Under the Receiver Operating Characteristic Curve) are essential, distinct metrics used to evaluate the performance of classification models in machine learning, particularly when accuracy is misleading due to imbalanced datasets.



For a dataset of this size, you should drop TfidfVectorizer entirely and switch to HashingVectorizer paired with a TfidfTransformer. This combination processes text instantly without saving a dictionary, easily handling 150k rows in seconds.



Scikit-Learn Gradient Boosting and RandomForestClassifier is slow when there is text involve and it is more suitable for small dataset. In this task, the dataset is 150k row. Therefore LogisticRegression, LinearSVC and LGBMClassifier model is used instead of Gradient Boosting and RandomForestClassifier.



The Modelling code is fine in MLP pipeline which explain why the code in EDA is different from the MLP pipeline as the code is being fine tune to generate ai summary. This can be seem only when the pipeline in run in UI interface to see the expected result and fine the code.





**Drop item**

Weighted similarity / semantic similarity comment output. Due my laptop performance issue, not able to use this method as it is too slow for I5 Gen4 laptop.

iterrows() Method is drop as it is too time consuming to loop through 150k rows one by one and it is not efficient.



\---

## 4\. Use of the AI assistant

Where did you use the AI assistant in this work? Give three specific examples of something the assistant suggested that you changed, rejected, or significantly modified — and explain your reasoning.



**Your answer:**



I will be using AI assistant to generate the code required based on the logic that I required. This will speed up the process for the task and I use my logic think requirement for AI assistant to generate the output I want as my logic analysis cannot be replace by AI. This is built based on my past work experience.



Three specific examples:

1. AI propose a simple ML pipeline that generate best F1 Score and ROC AUC. I decided to ask AI to provide me something more interactive so that user can input and get output that is where dash UI is propose.
2. AI propose certain code when I face certain coding issue but it is not correct answer. I have to provide the full code that has error then AI has provide me the correct answer.
3. AI propose solution that is very slow in generating eg code which read line by line. This is not efficient and slow so I tell I need something that is fast and do by clustering. I give the current code to AI and it is able to produce code that is much faster.







\---

## 5\. Next Steps

If you had one more week to continue this project, what would you do next, and why? What signals from your current work make those the right next steps?



**Your answer:**



I will do the following steps:

1. RAG Pipeline through the use of Agentic AI







**RAG Pipeline through the use of Agentic AI**



RAG Pipeline through the use of Agentic AI given that there is huge different data source (structure, unstructured, image) which can be use to gain insight. RAG Pipeline ML is a memory ML where the AI has database to look for and to prevent hallucinations and the AI can just go to web to search for information that Travelers’ Inn does not have.



User can ask the question in the UI website and give answer that can provide more insight instead of probability.

&#x20;





\---

