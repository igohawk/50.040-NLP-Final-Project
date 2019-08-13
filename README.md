# 50.040-NLP-Final-Project



## Part 5 Design Challenge
### How to run?

Download http://nlp.stanford.edu/data/glove.42B.300d.zip\
Create new `glove_path` folder in `part 5 design challenge` and unzip content into folder 

Before starting to run the code, make sure there are no `.p5.out` files in `data` and `test` folders (delete if any)

#### For training 
1. Open `cmd` 
2. `cd` into the `part 5 design challenge` folder
3. Open the `train.py` file and modify `train_lang = 'EN'` 
4. Run `python train.py` in `cmd`
5. Now open the `train.py` file again and modify `train_lang = 'ES'`
6. Run `python train.py` again
7. Notice that `EN_ner_model.h5`, `ES_ner_model.h5`, `tag2idx_EN.pickle`, `tag2idx_ES.pickle`, `word2idx_EN.pickle`, `word2idx_ES.pickle` files are created in the folder
8. Training is complete

#### For prediction
After training is complete, 
1. Open `predict.py` file and modify:\
 `main_path = 'data'`\
 `kind = 'dev'`\
 `test_lang = EN`
 2. Run `python predict.py` in `cmd`
 3. Open `predict.py` file again and this time modify `test_lang = 'ES'`
 4. Run `python predict.py` again
 5. Now prediction is done for the dev set and you should see the output `dev.p5.out` in both `data/EN` and `data/ES` folders respectively
 
 #### For evaluation
 After prediction is complete,
 1. Run `python Evaluation_Script\evalResult.py data\EN\dev.out data\EN\dev.p5.out` in`cmd`
 2. Results will be printed on screen
 
 #### For testing
 For testing of the test set, after training is complete,\
 (this procedure is similar to prediction except for the folder and file name) 
 1. Open `predict.py` file and modify:\
 `main_path = 'test'`\
 `kind = 'test'`\
 `test_lang = EN`
 2. Run `python predict.py` in `cmd`
 3. Open `predict.py` file again and this time modify `test_lang = 'ES'`
 4. Run `python predict.py` again
 5. Now prediction is done for the test set and you should see the output `test.p5.out` in both `test/EN` and `test/ES` folders respectively
