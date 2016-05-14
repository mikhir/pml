# Modeling weight lifting data

Weight lifting data consists of the classe variable and sensory information collected during the lifts. Classe variable is the one we are trying predict and it can take five different values - four of these indicate common mistake (B,C,D,E) in the exercise and one value indicates (A) the correct way of lifting. The data is divided to training and testing set, which are loaded into R below.
```
library(caret)
training <- read.csv('pml-training.csv')
testing <- read.csv('pml-testing.csv')
```
Looking at the data reveals that there are columns which should be removed before modeling. For example user_name should not be used for prediction because it is specific for this data and cannot be generalized. Same thing applies for the timestamps, the ID column X, num_window and new_window. Another problem with the data are the NA values which we can allow to some extent. The following piece of code removes the unwanted columns and removes columns which have 1000 or more NA values (~5% of total amount of values).
```
training <- training[!names(training) %in% c("X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window")]
training <- training[, colSums(is.na(training)) < 1000]
``` 
Removing the NA columns and other unwanted columns leaves us 86 columns which is quite a lot. To identify the important columns in the data we can use the varImp method of the caret package. This method takes a fitted model and determines which of the columns in the data count most of the variability in the predicted value. Thus, we must first fit a model to the data, to be able to detect the columns which have impact on the variability to the classe. Because this is a classification problem, decision tree seems to be reasonable choice for the modeling. The command below creates a decision tree with the caret package.

```r
set.seed(1)
fit <- train(classe~., training, method="rpart")
confusionMatrix(fit)
```

```
## Bootstrapped (25 reps) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 25.6 12.1 10.0 10.1  4.5
##          B  0.2  2.4  0.2  1.3  0.6
##          C  2.4  5.0  7.1  5.0  5.0
##          D  0.0  0.0  0.0  0.0  0.0
##          E  0.1  0.0  0.0  0.0  8.3
##                            
##  Accuracy (average) : 0.435
```
From the print above one can see that the accuracy of this individual decision tree is not that great (44%). One should realize however that even this result is lot better than just making a random selection for which the accuracy would be 20% (1/5). Furthermore, the accuracy can be improved with random forest method, but to run that we'll need to drop out columns which do not contribute to the classe variability for two reasons. First one is that random forest approach with 86 columns will take quite long to calculate and another one is to avoid overfitting. Let us run varImp to see which columns are important.

```r
varImp(fit)
```

```
## rpart variable importance
## 
##                   Overall
## roll_belt          100.00
## magnet_dumbbell_y   78.60
## accel_belt_z        61.15
## magnet_belt_y       57.99
## total_accel_belt    50.23
## roll_forearm        40.99
## magnet_arm_x        38.51
## accel_arm_x         37.42
## roll_arm            31.11
## magnet_dumbbell_z    0.00
## yaw_belt             0.00
## accel_dumbbell_y     0.00
## roll_dumbbell        0.00
```
From the above we can see that there 13 columns which contribute to the variability most. Let us remove other columns from the training set (still keeping classe obviously). We are also taking four columns which have variability 0 because in the final model we'll be fitting several models for which the variability might be bit different.
```
training <-training[,c("classe","roll_belt","magnet_dumbbell_y","accel_belt_z","magnet_belt_y","total_accel_belt","roll_forearm","magnet_arm_x","accel_arm_x","roll_arm","accel_dumbbell_y","roll_dumbbell","yaw_belt","magnet_dumbbell_z")]
```

Next, we can utilize the power of multiple decision trees by running random forest method. The command uses ten-fold cross-validation and enables parallel execution which will speed-up the processing because rf method can be easily run in parallel.

```r
betterFit <- train(classe~., data=training, method="rf", trControl=trainControl(method="cv",number=10), allowParallel=TRUE)
confusionMatrix(betterFit)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.1  0.0  0.0  0.0
##          B  0.1 19.1  0.2  0.0  0.0
##          C  0.0  0.2 17.2  0.2  0.0
##          D  0.0  0.0  0.1 16.1  0.1
##          E  0.0  0.0  0.0  0.0 18.2
##                             
##  Accuracy (average) : 0.9896
```
This model has very high accuracy, but on the downside it might be overfitted. But the reduction of data columns from 160 to 10 will of course leviate the problem of overfitting.
