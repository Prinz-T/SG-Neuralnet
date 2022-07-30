##---- Data import ----
cancer = read.csv(paste0("http://archive.ics.uci.edu/ml/machine-learning-databases/", "breast-cancer-wisconsin/breast-cancer-wisconsin.data"), header = FALSE, stringsAsFactors = F) # Load dataset from the UCI repository.
names(cancer) = c("ID", "thickness", "cell_size", "cell_shape", "adhesion", "epithelial_size", "bare_nuclei", "bland_cromatin", "normal_nucleoli", "mitoses", "class") # Add names to the dataset.
##---- Remove missing items and restore the outcome data ----
cancer = as.data.frame(cancer)
cancer$bare_nuclei = replace(cancer$bare_nuclei, cancer$bare_nuclei == "?",NA) # Recode missing values with NA.
cancer = na.omit(cancer) # Remove rows with missing values.
cancer$bare_nuclei <- as.numeric(cancer$bare_nuclei) # Change data type in number (TP).
write.table(cancer,"/path/Cancer.txt", sep = "\t", row.names = FALSE) # Write whole dataset (TP).
dim(cancer) # Dimensions of dataset (TP).
percentage <- prop.table(table(cancer$class)) * 100 # Summarize the class distribution (TP).
cbind(freq=table(cancer$class), percentage=percentage) # (TP).
summary(cancer) # Summarize attribute distributions (TP).
cancer$class = (cancer$class/2) - 1 # Recode the class (outcome) variable to 1 and 2.
head(cancer) # Show the first 6 rows of the dataset.
library(DataExplorer) # Load dataexlorer package into this R session (TP).
create_report(cancer) # Data profiling report as html file (TP).
##---- Data split ----
set.seed(080817) # Set a random seed so that repeated analyses have the same outcome. 
index = 1:nrow(cancer) # Create an index vector with as many sequential variables as there are.
testindex = sample(index, trunc(length(index)/3)) # Take a sample of 33.3% of the variable.
testset = cancer[testindex, ] # Create a test (validation) dataset with 33.3% of the data. 
trainset = cancer[-testindex, ] # Create a trainig dataset with 66.6% of the data.
write.table(testset,"/path/Cancer_testset.txt", sep = "\t", row.names = FALSE) # Write test dataset (TP).
write.table(trainset,"/path/Cancer_trainset.txt", sep = "\t", row.names = FALSE) # Write train dataset (TP).
x_train = data.matrix(trainset[, 2:10]) # Take the features (x) from the training dataset.
y_train = as.numeric(trainset[, 11]) # Take the outcomes (y) from the training dataset.
x_test = data.matrix(testset[, 2:10]) # Take the features (x) from the testing/validation dataset.
y_test = as.numeric(testset[, 11]) # Take the outcomes (y) from the testing/validation dataset.
##---- Model training ----
##---- Fit the ANN algorithm to the data and plot the nnet architecture ----
##install.packages("nnet")
require(nnet) # Load nnet package into this R session.
nnet_model = nnet(x_train, y_train, size=5) # Fit a single-layer neural network to the data with 5 units in the hidden layer.
require(NeuralNetTools) # Load the neuralnettools package into this R session (TP).
par(mar=c(5.1, 4.1, 4.1, 2.1)) # Setting plotting parameters (TP).
plotnet(nnet_model) # Plot the nnet model as neural interpretation diagram (TP).
neuralweights(nnet_model) # Get weights for nnet model (TP).
garson(nnet_model) # Analyzes relative importance of each variable through magnitude (TP).
##---- Model testing ----
##---- Extract predictions from the trained models on the new data ----
nnet_pred = round(predict(nnet_model, x_test, type="raw"),0) # Prediction vector for the neural network.
##---- Confusion matrix ----
require(caret) # Load the caret package into this R session.
confusionMatrix(as.factor(nnet_pred),as.factor(y_test)) # Create a confusion matrix for the neural network.
##---- ROC curve ----
require(pROC) # Load the pROC package into this R session.
roc_nnet = roc(as.vector(y_test), as.vector(nnet_pred))
plot.roc(roc_nnet, ylim=c(0,1), xlim=c(1,0)) # Plot the ROC curve.
lines(roc_nnet, col="blue")
legend("bottomright", legend=c("Neural Net"), col=c("blue"), lwd=2)
auc_nnet = auc(roc_nnet) # Calculate the area under the ROC curve.
print(auc_nnet) # Print the AUC value for NNET (TP)
##---- New data ----
thickness <- as.numeric(readline(prompt = "thickness?"))
cell_size <- as.numeric(readline(prompt = "cell size?"))
cell_shape <- as.numeric(readline(prompt = "cell shape?"))
adhesion <- as.numeric(readline(prompt = "adhesion?"))
epithelial_size <- as.numeric(readline(prompt = "epithelial size?"))
bare_nuclei <- as.numeric(readline(prompt = "bare nuclei?"))
bland_cromatin <- as.numeric(readline(prompt = "bland cromatin?"))
normal_nucleoli <- as.numeric(readline(prompt = "normal nucleoli?"))
mitoses <- as.numeric(readline(prompt = "mitoses?"))
new_data = c(thickness,cell_size,cell_shape,adhesion, epithelial_size,bare_nuclei,bland_cromatin,normal_nucleoli ,mitoses) # Combine the data.
new_pred_nnet = predict(nnet_model ,data.matrix(t(new_data)),type="raw") # Apply the new data to the validated nnet model.
if(round(new_pred_nnet, digits=0)>0) print("Result nnet model: Sample is malignant") else print("Result nnet model: Sample is benign") # Evaluating prediction nnet model (TP).
