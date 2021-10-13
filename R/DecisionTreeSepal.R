install.packages("sparklyr")
install.packages("rpart")
library(sparklyr)
library(rpart)

fit <- rpart(Sepal.Width~Sepal.Length + Sepal.Length + Petal.Width + Petal.Length + Species, method = "anova", data = iris)

png("decisionTree.png", width = 750, height = 750)

plot(fit, uniform = TRUE, main = "Decision Tree using Regression")

text(fit, use.n = TRUE, cex = .7)

dev.off()

print(fit)

ds <- data.frame(Species = 'versicolor',
                 Sepal.Length = 5.0,
                 Petal.Length = 4.8,
                 Petal.Width = 1.5)

cat("Predicted value :\n")
predict(fit, ds, method = "anova")
