# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py`
    - To run ML pipeline that trains classifier and saves(注意，这个过程在Udacity Workspace中花费约1小时)
        `python models/train_classifier.py`
    - 以上两个步骤在project's root directory里产生两个文件：
        `disaster_data.db`、`cv.pkl`
        
2. Run the following command in the project's root directory to run your web app.
    `python run.py`
    
3. Go to http://0.0.0.0:3001/

P.S. 本项目并没有上传对应的训练模型“cv.pkl”，因为文件超过300Mb，所以在repo push报错。
