import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.{Evaluator, RegressionEvaluator}
import org.apache.log4j._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

object usCleanData {
  def main(args: Array[String]):Unit={

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder()
      .master("local")
      .getOrCreate()

    val df = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load("..Clean_USA_Housing.csv")

    df.printSchema()

root
 |-- Avg Area Income: double (nullable = true)
 |-- Avg Area House Age: double (nullable = true)
 |-- Avg Area Number of Rooms: double (nullable = true)
 |-- Avg Area Number of Bedrooms: double (nullable = true)
 |-- Area Population: double (nullable = true)
 |-- Price: double (nullable = true)

    val sd = df.select(col("Price").as("label"), col("Avg Area Income"), col("Avg Area House Age"), col("Avg Area Number of Rooms"), col("Avg Area Number of Bedrooms"), col("Area Population"))

    sd.show(10)
    
+------------------+------------------+------------------+------------------------+---------------------------+------------------+
|             label|   Avg Area Income|Avg Area House Age|Avg Area Number of Rooms|Avg Area Number of Bedrooms|   Area Population|
+------------------+------------------+------------------+------------------------+---------------------------+------------------+
|1059033.5578701235| 79545.45857431678| 5.682861321615587|       7.009188142792237|                       4.09|23086.800502686456|
|  1505890.91484695| 79248.64245482568|6.0028998082752425|       6.730821019094919|                       3.09| 40173.07217364482|
|1058987.9878760849|61287.067178656784| 5.865889840310001|       8.512727430375099|                       5.13| 36882.15939970458|
|1260616.8066294468| 63345.24004622798|7.1882360945186425|       5.586728664827653|                       3.26| 34310.24283090706|
| 630943.4893385402| 59982.19722570803| 5.040554523106283|       7.839387785120487|                       4.23|26354.109472103148|
|1068138.0743935304|  80175.7541594853|4.9884077575337145|       6.104512439428879|                       4.04|26748.428424689715|
|1502055.8173744078| 64698.46342788773| 6.025335906887152|       8.147759585023431|                       3.41| 60828.24908540716|
|1573936.5644777215| 78394.33927753085|6.9897797477182815|       6.620477995185026|                       2.42|36516.358972493836|
| 798869.5328331633| 59927.66081334963|  5.36212556960358|      6.3931209805509015|                        2.3| 29387.39600281585|
|1545154.8126419624| 81885.92718409566| 4.423671789897876|       8.167688003472351|                        6.1| 40149.96574921337|
+------------------+------------------+------------------+------------------------+---------------------------+------------------+
only showing top 10 rows

    val assembler = new VectorAssembler()
      .setInputCols(Array("Avg Area Income", "Avg Area House Age", "Avg Area Number of Rooms", "Avg Area Number of Bedrooms", "Area Population"))
      .setOutputCol("features")

    val output = assembler.transform(sd)
      .select(col("label"), col("features"))

    val Array(training, test) = output
      .select("label", "features")
      .randomSplit(Array(0.7, 0.3), seed=1111)

    val lr = new LinearRegression()

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10000, 0.1))
      .build()

    val trainValidationSplit = (new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8))

    val model = trainValidationSplit.fit(training)

    model.transform(test).select("features", "label", "Prediction").show()

}
}

    
+--------------------+------------------+------------------+
|            features|             label|        Prediction|
+--------------------+------------------+------------------+
|[37971.2075662352...|31140.517620186045|108186.87186273467|
|[40366.6162912572...|152071.87474956046|224340.32816796517|
|[50926.7766338627...|211017.97049475575| 472003.9693052508|
|[46367.2058588838...|268050.81474351394|266737.39937116625|
|[44688.5638164431...|291724.24561034393| 405705.1974021299|
|[35797.3231215482...| 299863.0401311839| 382286.0026657176|
|[41240.0572765673...|319495.66759175994|472741.22141373996|
|[52511.6543462467...| 325195.9428324207| 428538.0878053298|
|[56851.9957053819...|393639.07395721273| 564924.4782571881|
|[71721.4213772300...| 395440.2021544269| 447414.2732003755|
|[54465.7473655343...|395523.24608349975|  374584.741259967|
|[39653.7700307378...| 395901.2500673755| 539768.8874357063|
|[66469.3694730564...| 412269.2033995612| 659983.8663477572|
|[35608.9862370775...| 449331.5835333807|  550080.582269974|
|[49736.1420936821...| 465499.8080598057|494741.85629320657|
|[59314.7884759627...|470008.13822756393|508084.32774383016|
|[62639.1598082764...| 476971.4559427722| 718465.7872525472|
|[52723.8765553883...| 479500.5568108269| 620147.9159923783|
|[55557.7401920089...|499548.01143053523|  534323.134850408|
|[42822.3110961679...| 513215.9882314093|438438.39507413376|
+--------------------+------------------+------------------+
only showing top 20 rows


