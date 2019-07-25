# R script to visualize the v score (return % per trading day)
# for different trader models
# for training and testing data set
# as a horizontal bar plot
#
# run with: Rscript model-performance.R
# input: model-performance.csv
#        (data collected manually from output of
#         multiple runs of deep_q_learning_trader.py for different dql model parameters
#         and from running stock_exchange.py with main_print_model_performance() once to get the data for
#         the reference traders)
# outputs: model-performance.svg

data <- read.csv(file="model-performance.csv", sep=";", row.names=NULL, header=TRUE)
data_train <- data[data$dataset == "train", ]
data_test  <- data[data$dataset == "test" , ]
svg(file="model-performance.svg", width=10, height=6)

par(mar=c(5,11,1,1))  # default: 5,4,4,2 -> increase space for y axis labels
m <- matrix(
    c(rev(data_test$returnpctperday), rev(data_train$returnpctperday)),
    nrow=length(data_test$returnpctperday), ncol=2, byrow=FALSE,
    dimnames=list(rev(data_test$trader), c("test", "train"))
)

barplot(
    t(m),
    # names.arg=data_test$trader,
    # main="Trader Model Performance",
    # col=c('green4', 'green2'),
    col=c('steelblue', '#d1dfed'), # 'slategray2'),
    horiz=TRUE,
    border=NA,
    beside=TRUE,
    xlab="", # "Return per day in percent",
    ylab="",
    xlim=c(0, max(m)*1.2),
    las=1,  # horizontal axis labels
    legend.text=rev(c("Training", "Testing")),
)
abline(
    v=0.357,
    lty=3,
    col='gray60'
)
mtext("Return per day in percent", side=1, line=3, adj=0.5, at=0.3)
dev.off()
