# NYPD call assistant

---

This project of the part of requirement task at Data Science club at PJATK. 

We were provided with NYPD dataset, with ~9 millions rows, model that we needed to train was up to our decision, so we decided to make Call assistant for 911 call center. 

## Steps to run this project
Do step by step 

1. run command `python main.py download_data` 
2. run command `python main.py preprocess_data`
3. run command `python train_model`
4. run command `streamlit run app.py`

## Problems

Because of luck of time, we have created the POS, rather than well-structured project, we had to sacrifice model accuracy, quality of GUI and data cleaning steps were not as thorough as we would like(probably every aspect of the project has space to improve)


## Steps taken during the development

1. **Downloading the data:**  that were size of 3.7Gb was not that trivial especially when max size we have worked with has 10Mb csv file. To solve time taken by downloading we used multithreading, and this speed up the process a lot. To reduce memory issue, when downloading the data we added preprocess steps, dropped all the columns that were useless for our task, so in the end csv file from 3.7Gb became 700 Mb
2. **Preprocess steps:** dataset originally had 36 columns + 4 from API, we managed reduce this number up to 10 features(one feature also correlated with another, but we had to time to rewrite GUI because of this) + 1 label column. Labels also were a pain in the ***. They were unclean, with using similar names such as `LOITERING/GAMBLING` and `GAMBLING`
3. **Training the model:** we used DataIku to see what models are best for your task, we could use AutoGluon if accuracy were crucial, but it were not our case
4. **GUI:** we used streamlit to make user interface, we could push app into Streamlit Share, but it will take more time to rewrite it  
## Results

After preprocessing of the labels, we had 25 labels to classify, instead of 73. And without any fine-tuning we archived 41% accuracy, of course we didn't use cross-validation, and this result most likely will be worse in reality




