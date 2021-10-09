
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions.col

object TitanicKaggle {
  def main(args: Array[String]):Uni t= {
  
    // creatring spark session
    
    val spark = SparkSession
      .builder()
      .appName("mlPratice")
      .master("local[*]")
      .getOrCreate()

    // importing the data into the spark session

    val df = spark.read.option("header", "true")
      .option("inferschema", "true")
      .format("csv")
      .load("/titanic_train.csv")

    df.show(10)
    df.printSchema()
    
    // first view to check for the sample data

    df.createOrReplaceTempView("d1")
   
    val sd = spark.sql("SELECT survived, Pclass, Name, Sex, Age, SibSp, Parch, Fare, Embarked FROM d1")
    
    // drop  the null values
    
    val nd = sd.na.drop()
    nd.show(10)
    
    // coverting the Sex and Emabarked columns to int to process
    
    val x = new StringIndexer().setInputCol("Sex").setOutputCol("SexVector")
    val y = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedVector")

    val x1 = new OneHotEncoder().setInputCol("SexVector").setOutputCol("SexChanged")
    val y1 = new OneHotEncoder().setInputCol("EmbarkedVector").setOutputCol("EmbarkedChanged")

    val assembler = (new VectorAssembler()
      .setInputCols(Array("survived", "SexChanged", "EmbarkedChanged", "Age", "SibSp", "Parch", "Fare"))
      .setOutputCol("features"))

    val Array(training, test) = nd.randomSplit(Array(0.7, 0.3), seed = 55)

    import org.apache.spark.ml.Pipeline

    val lr = new LogisticRegression()
    
    // putting all the encoders and the assemblers in a pipeline
    
    val pipeline = new Pipeline().setStages(Array(x, y, x1, y1, assembler, lr))

    val model = pipeline.fit(training)

    val res = model.transform(test)

    res.show(10)

    res.printSchema()
    
    // exporting the df to a json file to process in RStudio for visualization
    
    res.coalesce(1).write.format("json").save("titanicMlResult")

  }
}
