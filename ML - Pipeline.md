### `Define the stages of the Pipeline`

  * **STAGE 1**: [Transformer] Fill null values with in each column
  * **STAGE 2**: [Transformer] Reduce Category
  * **STAGE 3**: [Estimator] Label Encode: Traffic
  * **STAGE 4**: [Estimator] Label Encode: Country
  * **STAGE 5**: [Estimator] Label Encode: Browser
  * **STAGE 6**: [Estimator] Label Encode: OS
  * **STAGE 7**: [Estimator] OHE: Country, Browser, OS
  * **STAGE 8**: [Transformer] Create Feature: Total Click per Publisher ID, Total Click per Campaign ID
  * **STAGE 9**: Transformer] Create Vector [ Traffic (LE), Fraud, Average_Click per Puplisher ID, Average Click per Campaign ID, Country (OHE), Browser (OHE), Device (OHE), OS (OHE)]
  * **STAGE 10**: [Estimator] Predict Labels Using the Logistic Regression

'''python
# custom transformer to fill null values

class nullValuesTransformer(Transformer):
    
    def __init__(self, dataframe = None):
        self.dataframe = dataframe
    
    def _transform(self, dataframe):
        dataframe = dataframe.fillna({
            "Fraud" : 0,
            "Country": "IN",
            "TrafficType" : "U",
            "Device": "Generic",
            "Browser": "chrome",
            "OS": "Android",
        })
        
        return dataframe
'''
'''
# Creating custom Transformer to reduce the categories of multiple features
class reduceCategories(Transformer):
    
    def __init__(self, dataframe = None):
        self.dataframe = dataframe
        
    def _transform(self, dataframe):
        
        # map the countries
        dataframe = dataframe.withColumn("country_modified", map_countries_udf(dataframe["Country"]))
        
         # map device
        dataframe = dataframe.withColumn("device_modified", map_device_udf(dataframe["Device"]))
        
        # map browser
        dataframe = dataframe.withColumn("browser_modified", map_browser_udf(dataframe["Browser"]))

        # map the os
        dataframe = dataframe.withColumn("os_modified", map_os_udf(dataframe["OS"]))
        
        return dataframe

# Creating two new features: total clicks per campaign Id, total clicks per publisher Id
class frequencyEncoding(Transformer):
    
    def __init__(self, dataframe = None):
        self.dataframe = dataframe
        
    def _transform(self, dataframe):
        
        # join total clicks per advertiser Campaign Id
        dataframe = dataframe.join(total_c_id, on="advertiserCampaignId")
        
        # join total clicks per publisher id dataframe
        dataframe = dataframe.join(total_p_id, on="publisherId")
        
        # replace null values
        dataframe = dataframe.fillna({
            'total_p_id': 0.0,
            'total_c_id' : 0.0,
        })
        
        return dataframe

# Stage 1 - replace null values
stage_1 = nullValuesTransformer()

# stage 2 - reduce categories
stage_2 = reduceCategories()

# Stage 3 - label encode Traffic_Type column
stage_3 = StringIndexer(inputCol= "TrafficType", outputCol= "traffic_le") 

# Stage 4 - label encode Country column
stage_4 = StringIndexer(inputCol= "country_modified", outputCol= "country_le")

# Stage 5 - label encode Browser column
stage_5 = StringIndexer(inputCol= "browser_modified", outputCol= "browser_le")

# Stage 6 - label encode OS column
stage_6 = StringIndexer(inputCol= "os_modified", outputCol= "os_le")

 # Stage 7 - One Hot Encode columns
stage_7 = OneHotEncoder(inputCols= ["country_le",  "browser_le", "os_le", "traffic_le"], 
                        outputCols= ["country_ohe",  "browser_ohe", "os_ohe", "traffic_ohe"])

# stage 8 - Create new features for total clicks per campaign id and per pulisher id
stage_8 = frequencyEncoding()

# Stage 9 - Create vector from the columns
stage_9 = VectorAssembler(inputCols= ["Fraud",
                                      "traffic_ohe",
                                      "country_ohe",
                                      "device_modified",
                                      "browser_ohe",
                                      "os_ohe",
                                      'total_p_id',
                                      'total_c_id',],

                         outputCol=  "feature_vector")

# Stage 10 - Train ML model
stage_10 = classification.LogisticRegression(featuresCol= "feature_vector", labelCol= "ConversionStatus")

# Define pipeline
pipeline = Pipeline(stages= [stage_1,
                             stage_2,
                             stage_3,
                             stage_4,
                             stage_5,
                             stage_6,
                             stage_7,
                             stage_8,
                             stage_9,
                             stage_10])

# fit the pipeline with the training data
pipeline_model = pipeline.fit(train_data_pipeline)

# transform data
final_data = pipeline_model.transform(train_data_pipeline)

final_data.select("ID", "ConversionStatus", "rawPrediction", "probability", "prediction").show(10)

final_valid_data = pipeline_model.transform(valid_data_pipeline)
final_valid_data.columns

final_valid_data.select("ID", "ConversionStatus", "rawPrediction", "probability", "prediction").show(10)

'''