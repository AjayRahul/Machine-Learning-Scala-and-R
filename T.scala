package com.sundogsoftware.spark
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions.col

object T {
  def main(args: Array[String]):Unit= {
    val spark = SparkSession
      .builder()
      .appName("mlPratice")
      .master("local[*]")
      .getOrCreate()

    // COMMAND ----------

    val df = spark.read.option("header", "true")
      .option("inferschema", "true")
      .format("csv")
      .load("C:\\Users\\Lenovo\\Documents\\SparkScalaCourse\\SparkScalaCourse\\IdeaProjects\\Ttitanic\\titanic_train.csv")

    df.show(10)
    df.printSchema()

    df.createOrReplaceTempView("d1")
    val sd = spark.sql("SELECT survived as label, Pclass, Name, Sex, Age, SibSp, Parch, Fare, Embarked FROM d1")

    val nd = sd.na.drop()
    nd.show(10)

    val x = new StringIndexer().setInputCol("Sex").setOutputCol("SexVector")
    val y = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedVector")

    val x1 = new OneHotEncoder().setInputCol("SexVector").setOutputCol("SexChanged")
    val y1 = new OneHotEncoder().setInputCol("EmbarkedVector").setOutputCol("EmbarkedChanged")

    val assembler = (new VectorAssembler()
      .setInputCols(Array("Pclass", "SexChanged", "EmbarkedChanged", "Age", "SibSp", "Parch", "Fare"))
      .setOutputCol("features"))

    val Array(training, test) = nd.randomSplit(Array(0.7, 0.3), seed = 55)

    import org.apache.spark.ml.Pipeline

    val lr = new LogisticRegression()

    val pipeline = new Pipeline().setStages(Array(x, y, x1, y1, assembler, lr))

    val model = pipeline.fit(training)

    val res = model.transform(test)

    res.show(10)

    res.printSchema()

    res.coalesce(1).write.format("json").save("titanicMlResult")

  }
}
