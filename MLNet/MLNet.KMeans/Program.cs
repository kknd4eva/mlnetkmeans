// -----------------------------------------------------------
// File: Program.cs      dotnet add package Microsoft.ML
//                       dotnet run
// -----------------------------------------------------------
using Amazon.DynamoDBv2;
using Amazon.DynamoDBv2.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Text.Json;

namespace MLNet.KMeans;

public class SteamMediaData
{
    [LoadColumn(0)] public int SteamAppId { get; set; }
    [LoadColumn(1)] public string HeaderImage { get; set; } = string.Empty;
    [LoadColumn(2)] public string ScreenshotsJson { get; set; } = string.Empty;
    [LoadColumn(3)] public string Background { get; set; } = string.Empty;
    [LoadColumn(4)] public string MoviesJson { get; set; } = string.Empty;
}

public class MediaScreenshot
{
    public int id { get; set; }
    public string path_thumbnail { get; set; } = string.Empty;
    public string path_full { get; set; } = string.Empty;
}

public class SteamGameCSV
{
    [LoadColumn(0)] public int AppId { get; set; }
    [LoadColumn(1)] public string Name { get; set; } = string.Empty;
    [LoadColumn(2)] public string ReleaseDate { get; set; } = string.Empty;
    [LoadColumn(3)] public float English { get; set; }
    [LoadColumn(4)] public string Developer { get; set; } = string.Empty;
    [LoadColumn(5)] public string Publisher { get; set; } = string.Empty;
    [LoadColumn(6)] public string Platforms { get; set; } = string.Empty;
    [LoadColumn(7)] public float RequiredAge { get; set; }
    [LoadColumn(8)] public string Categories { get; set; } = string.Empty;
    [LoadColumn(9)] public string Genres { get; set; } = string.Empty;
    [LoadColumn(10)] public string SteamSpyTags { get; set; } = string.Empty;
    [LoadColumn(11)] public float Achievements { get; set; }
    [LoadColumn(12)] public float PositiveRatings { get; set; }
    [LoadColumn(13)] public float NegativeRatings { get; set; }
    [LoadColumn(14)] public float AveragePlaytime { get; set; }
    [LoadColumn(15)] public float MedianPlaytime { get; set; }
    [LoadColumn(16)] public string Owners { get; set; } = string.Empty;
    [LoadColumn(17)] public float Price { get; set; }
}

public class SteamTagData
{
    [LoadColumn(0)] public int AppId { get; set; }
    [LoadColumn(1, 371)] public float[] TagValues { get; set; } = Array.Empty<float>();
}

public class GameData
{
    public int AppId { get; set; }
    public string Title { get; set; } = string.Empty;
    public DateTime ReleaseDate { get; set; }
    public float PositiveRatio { get; set; }
    public float Price { get; set; }
    public bool IsWin { get; set; }
    public bool IsMac { get; set; }
    public bool IsLinux { get; set; }
    [VectorType(371)]
    public float[] TagVector { get; set; } = Array.Empty<float>();
    public string ImageUrl { get; set; } = string.Empty; 
}


/// <summary>Prediction type for K-Means</summary>
public class GameCluster
{
    [ColumnName("PredictedLabel")]
    public uint ClusterId { get; set; }

    public float[] Distances { get; set; } = Array.Empty<float>();
}

public class DynamoDbExportRecord
{
    public int AppId { get; set; }
    public uint ClusterId { get; set; }
    public float DistanceToCentroid { get; set; }
    public string ImageUrl { get; set; } = string.Empty;
    public string ThumbnailUrl { get; set; } = string.Empty;
    public string GameJson { get; set; } = string.Empty;
}

public class Program
{
    public static async Task Main(string[] args)
    {
        var engine = new GameClusterEngine();
        engine.LoadAndPrepareData("data/steam.csv", "data/steamspy_tag_data.csv", "data/steam_media_data.csv");
        engine.TrainModel(12);
        engine.EvaluateModel();
        engine.SaveModel("data/model.zip");

        // Prepare records for DynamoDB export
        var exportRecords = engine.PrepareDynamoDbExport(false);
        Console.WriteLine($"Prepared {exportRecords.Count} records for DynamoDB export.");

        // Write to DynamoDB table 'GameData'
        using var client = new AmazonDynamoDBClient();
        foreach (var record in exportRecords)
        {
            var item = new Dictionary<string, AttributeValue>
            {
                ["app_id"] = new AttributeValue { N = record.AppId.ToString() },
                ["cluster_id"] = new AttributeValue { N = record.ClusterId.ToString() },
                ["distance_to_centroid"] = new AttributeValue { N = record.DistanceToCentroid.ToString("F4") },
                ["image_url"] = new AttributeValue { S = record.ImageUrl },
                ["game_json"] = new AttributeValue { S = record.GameJson }
            };
            var request = new PutItemRequest
            {
                TableName = "GameClusters",
                Item = item
            };
            await client.PutItemAsync(request);
        }

        Console.WriteLine("All records written to DynamoDB table 'GameData'.");

        Console.WriteLine("\nGetting 5 recommendations for app_id 320 (Half Life 2)…");
        foreach (var g in engine.GetSimilarGames(320, 5))
            Console.WriteLine($"• {g.Title} (id={g.AppId})");
    }
}

public sealed class GameClusterEngine
{
    private readonly MLContext _ml = new(seed: 42);
    private ITransformer? _model;
    private List<GameData> _games = new();
    private Dictionary<int, SteamMediaData> _mediaDict = new();


    // ------------- DATA LOADING ------------------------------------------------
    public void LoadAndPrepareData(string mainCsvPath, string tagCsvPath, string mediaCsvPath)
    {
        var mainView = _ml.Data.LoadFromTextFile<SteamGameCSV>(mainCsvPath, hasHeader: true, separatorChar: ',');
        var tagView = _ml.Data.LoadFromTextFile<SteamTagData>(tagCsvPath, hasHeader: true, separatorChar: ',');

        // 1) Load media data into a lookup dictionary
        var mediaView = _ml.Data.LoadFromTextFile<SteamMediaData>(
            mediaCsvPath, hasHeader: true, separatorChar: ',');
        _mediaDict = _ml.Data
                      .CreateEnumerable<SteamMediaData>(mediaView, reuseRowObject: false)
                      .ToDictionary(m => m.SteamAppId);

        // Create a lookup for tag vectors by AppId
        var tagDict = _ml.Data.CreateEnumerable<SteamTagData>(tagView, reuseRowObject: false)
                              .ToDictionary(x => x.AppId, x => x.TagValues);

        // Join main data with tag data by AppId, filter out games without tags
        _games = _ml.Data.CreateEnumerable<SteamGameCSV>(mainView, reuseRowObject: false)
                   .Where(game => tagDict.ContainsKey(game.AppId))
                   .Select(game => new GameData
                   {
                       AppId = game.AppId,
                       Title = game.Name,
                       ReleaseDate = DateTime.TryParse(game.ReleaseDate, out var dt) ? dt : DateTime.MinValue,
                       PositiveRatio = (game.PositiveRatings + game.NegativeRatings) > 0
                                       ? game.PositiveRatings / (game.PositiveRatings + game.NegativeRatings)
                                       : 0f,
                       Price = game.Price,
                       IsWin = game.Platforms.ToLower().Contains("windows"),
                       IsMac = game.Platforms.ToLower().Contains("mac"),
                       IsLinux = game.Platforms.ToLower().Contains("linux"),
                       TagVector = tagDict[game.AppId].Select(v => v > 0 ? 1f : 0f).ToArray()
                   })
                   .ToList();
        }
    

    // ------------- PIPELINE ----------------------------------------------------
    public void TrainModel(int numberOfClusters)
    {
        var data = _ml.Data.LoadFromEnumerable(_games);
        Console.WriteLine($"Number of games: {_games.Count}");

        var pipeline = _ml.Transforms
            .NormalizeMinMax(nameof(GameData.PositiveRatio))
            .Append(_ml.Transforms.NormalizeMinMax(nameof(GameData.Price)))
            .Append(_ml.Transforms.Conversion.ConvertType(nameof(GameData.IsWin), outputKind: DataKind.Single))
            .Append(_ml.Transforms.Conversion.ConvertType(nameof(GameData.IsMac), outputKind: DataKind.Single))
            .Append(_ml.Transforms.Conversion.ConvertType(nameof(GameData.IsLinux), outputKind: DataKind.Single))
            .Append(_ml.Transforms.Concatenate("Features",
                nameof(GameData.PositiveRatio),
                nameof(GameData.Price),
                nameof(GameData.IsWin),
                nameof(GameData.IsMac),
                nameof(GameData.IsLinux),
                nameof(GameData.TagVector)))
            .Append(_ml.Clustering.Trainers.KMeans("Features", numberOfClusters: numberOfClusters));

        Console.WriteLine("⏳ training …");
        _model = pipeline.Fit(data);
        Console.WriteLine($"✅ model trained ({numberOfClusters} clusters).");
    }


    // ------------- EVALUATION --------------------------------------------------
    public void EvaluateModel()
    {
        if (_model is null) throw new InvalidOperationException("Train the model first.");

        var dv = _ml.Data.LoadFromEnumerable(_games);
        var scored = _model.Transform(dv);
        var metrics = _ml.Clustering.Evaluate(scored);

        Console.WriteLine($"\nClustering Evaluation Metrics:");
        Console.WriteLine($"  Average Distance     : {metrics.AverageDistance:F4}");
        Console.WriteLine($"  Davies-Bouldin Index : {metrics.DaviesBouldinIndex:F4}");


        // simple cluster histogram
        var predEng = _ml.Model.CreatePredictionEngine<GameData, GameCluster>(_model);
        var counts = _games.GroupBy(g => predEng.Predict(g).ClusterId)
                             .ToDictionary(g => g.Key, g => g.Count());
        Console.WriteLine("\nCluster distribution:");
        foreach (var (id, c) in counts.OrderBy(k => k.Key))
            Console.WriteLine($"  cluster {id,2}: {c} games");
    }

    // ------------- SIMILARITY LOOK-UP -----------------------------------------
    public List<GameData> GetSimilarGames(int appId, int k = 6)
    {
        if (_model is null) throw new InvalidOperationException("Train the model first.");

        var target = _games.FirstOrDefault(g => (int)g.AppId == appId);
        if (target is null)
        {
            Console.WriteLine($"⚠️ No game found with AppId {appId}.");
            return [];
        }

        var engine = _ml.Model.CreatePredictionEngine<GameData, GameCluster>(_model);
        var tgtPred = engine.Predict(target);

        // Apply transformations from the model pipeline before accessing Features column
        var data = _ml.Data.LoadFromEnumerable(_games);
        var transformedData = _model.Transform(data);
        var feats = transformedData.GetColumn<VBuffer<float>>("Features").ToArray();

        int idxT = _games.IndexOf(target);
        var tgtVec = feats[idxT];

        static float Cosine(VBuffer<float> a, VBuffer<float> b)
        {
            var da = a.DenseValues().ToArray();
            var db = b.DenseValues().ToArray();
            float dot = 0, na = 0, nb = 0;
            for (int i = 0; i < da.Length; i++)
            { dot += da[i] * db[i]; na += da[i] * da[i]; nb += db[i] * db[i]; }
            return dot / (float)(Math.Sqrt(na) * Math.Sqrt(nb) + 1e-6);
        }

        var candidates =
            _games.Select((g, i) => (game: g, idx: i))
                  .Where(t => (int)t.game.AppId != appId &&
                               engine.Predict(t.game).ClusterId == tgtPred.ClusterId)
                  .Select(t => (t.game, sim: Cosine(tgtVec, feats[t.idx])))
                  .OrderByDescending(x => x.sim)
                  .Take(k)
                  .Select(x => x.game)
                  .ToList();

        return candidates;
    }

    // ------------- SAVE / LOAD -------------------------------------------------
    public void SaveModel(string path)
    {
        if (_model is null) throw new InvalidOperationException("Train the model first.");
        _ml.Model.Save(_model, _ml.Data.LoadFromEnumerable(_games).Schema, path);
        Console.WriteLine($"📦 model saved → {path}");
    }

    public List<DynamoDbExportRecord> PrepareDynamoDbExport(bool isTest = false)
    {
        if (_model is null)
            throw new InvalidOperationException("Train the model first.");

        var exportList = new List<DynamoDbExportRecord>();
        var data = _ml.Data.LoadFromEnumerable(_games);
        var scoredData = _model.Transform(data);

        var clusterIdCol = scoredData.GetColumn<uint>("PredictedLabel").ToArray();
        var distancesCol = scoredData.GetColumn<float[]>("Score").ToArray();

        // if test mode, limit to first 10 games
        if (isTest)
        {
            clusterIdCol = clusterIdCol.Take(10).ToArray();
            distancesCol = distancesCol.Take(10).ToArray();
            _games = _games.Take(10).ToList();
        }

        for (int i = 0; i < _games.Count; i++)
        {
            Console.WriteLine($"Processing game: {_games[i].Title} (id={_games[i].AppId})");
            var game = _games[i];
            var clusterId = clusterIdCol[i];

            // Skip invalid or unassigned cluster (clusterId == 0)
            if (clusterId == 0)
                continue;

            var distances = distancesCol[i];
            float distanceToCentroid;

            // Score vector uses zero-based indexing: Score[k] = distance to cluster k
            if (clusterId < distances.Length)
                distanceToCentroid = distances[clusterId];
            else
                distanceToCentroid = float.MaxValue;

            // Lookup media
            _mediaDict.TryGetValue(game.AppId, out var md);
            var imageUrl = md?.HeaderImage ?? string.Empty;

            // Serialize game attributes to JSON for storage
            var gameJsonObj = new
            {
                game.AppId,
                game.Title,
                game.ReleaseDate,
                game.PositiveRatio,
                game.Price,
                game.IsWin,
                game.IsMac,
                game.IsLinux,
                game.TagVector
            };

            exportList.Add(new DynamoDbExportRecord
            {
                AppId = game.AppId,
                ClusterId = clusterId,
                DistanceToCentroid = distanceToCentroid,
                ImageUrl = imageUrl,
                GameJson = JsonSerializer.Serialize(gameJsonObj)
            });
        }

        return exportList;
    }
}
