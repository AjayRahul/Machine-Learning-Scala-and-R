import org.apache.spark.sql.SparkSession
import org.apache.log4j._

object KMeansMlExercise {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel("Level.ERROR")

    val spark = SparkSession.builder().master("local[*]").getOrCreate()

    val df = spark.read.format("csv").load("/Wholesale customers data.csv")

    val kmeans = new Kmeans().setK(2).setSeed(1L)
    val model = kmeans.fit(df)
    val WESS = model.computeCost(df)
    
    model.clusterCenter.foreach(println)
  }
}
